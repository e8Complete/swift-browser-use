# backend/main.py

import asyncio
import base64
import logging
import os
import uuid
from typing import Dict, Optional, Any

import uvicorn
from browser_use import Agent, Browser, BrowserConfig
from browser_use.agent.views import AgentHistoryList, AgentState, BrowserStateHistory # Import necessary views
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

# --- Load Environment Variables ---
# Ensure this runs before other imports that might depend on env vars
load_dotenv()

# --- Basic Logging Setup ---
# Configure logging level (consider using BROWSER_USE_LOGGING_LEVEL if set)
log_level_str = os.getenv("BROWSER_USE_LOGGING_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level_str, logging.INFO))
logger = logging.getLogger(__name__)


# --- LLM Instantiation Helper ---
def get_llm_instance() -> BaseChatModel:
    """Creates a LangChain LLM instance based on environment variables."""
    provider = os.getenv("AGENT_LLM_PROVIDER", "openai").lower()
    model_name = os.getenv("AGENT_LLM_MODEL")

    logger.info(f"Configuring LLM Provider: {provider}, Model: {model_name}")

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            if not os.getenv("OPENAI_API_KEY"): raise ValueError("OPENAI_API_KEY not set")
            if not model_name: raise ValueError("AGENT_LLM_MODEL not set for openai")
            return ChatOpenAI(model=model_name, temperature=0)

        elif provider == "azure":
             from langchain_azure_openai import AzureChatOpenAI
             if not os.getenv("AZURE_ENDPOINT"): raise ValueError("AZURE_ENDPOINT not set")
             if not os.getenv("AZURE_OPENAI_API_KEY"): raise ValueError("AZURE_OPENAI_API_KEY not set")
             if not model_name: raise ValueError("AGENT_LLM_MODEL (Azure deployment name) not set")
             api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
             return AzureChatOpenAI(
                 azure_deployment=model_name,
                 azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                 api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                 api_version=api_version,
                 temperature=0
             )

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            if not os.getenv("ANTHROPIC_API_KEY"): raise ValueError("ANTHROPIC_API_KEY not set")
            if not model_name: model_name = "claude-3-sonnet-20240229"
            return ChatAnthropic(model=model_name, temperature=0)

        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            if not os.getenv("GEMINI_API_KEY"): raise ValueError("GEMINI_API_KEY not set")
            if not model_name: model_name = "gemini-1.5-flash-latest"
            return ChatGoogleGenerativeAI(model=model_name, temperature=0, convert_system_message_to_human=True)

        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            if not model_name: raise ValueError("AGENT_LLM_MODEL not set for ollama")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            logger.info(f"Connecting to Ollama at {base_url} with model {model_name}")
            return ChatOllama(model=model_name, base_url=base_url, temperature=0)

        else:
            raise ValueError(f"Unsupported AGENT_LLM_PROVIDER: {provider}")

    except ImportError as e:
        logger.error(f"Missing LangChain package for provider '{provider}': {e}. Please install it (e.g., `uv pip install langchain-{provider}` or similar).")
        raise HTTPException(status_code=501, detail=f"LLM provider '{provider}' integration not installed.")
    except ValueError as e:
        logger.error(f"Configuration error for LLM provider '{provider}': {e}")
        raise HTTPException(status_code=500, detail=f"LLM configuration error: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM: {e}")


# --- FastAPI App Setup ---
app = FastAPI(title="Swift Agent Backend")

# --- CORS Configuration ---
# Make sure frontend URLs are correctly listed
origins = [
    "http://localhost:3000", # Default Next.js dev port
    # Add your deployed frontend URL here, e.g.:
    # "https://your-swift-deployment.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- Agent Management State ---
# Using simple dictionaries for state. Consider Redis/DB for production scalability.
agents: Dict[str, Agent] = {}
agent_status: Dict[str, str] = {} # e.g., "idle", "running", "done_success", "done_failure", "error"
websockets: Dict[str, WebSocket] = {}
agent_tasks: Dict[str, asyncio.Task] = {} # Stores the asyncio task running the agent

# --- Shared Browser Instance ---
# Configure Browser based on environment if needed
# e.g., BROWSER_HEADLESS=true, BROWSER_DISABLE_SECURITY=true etc.
browser_headless = os.getenv("BROWSER_HEADLESS", "False").lower() in ('true', '1', 't')
# Add more browser config vars as needed
shared_browser = Browser(config=BrowserConfig(headless=browser_headless))

# --- Pydantic Models for API ---
class TaskRequest(BaseModel):
    task: str
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    session_id: str
    status: str
    message: str

# --- WebSocket Helper ---
async def send_update(session_id: str, data: Dict[str, Any]):
    """Helper function to send JSON updates over WebSocket."""
    ws = websockets.get(session_id)
    if ws:
        try:
            await ws.send_json(data)
            # logger.debug(f"WS Sent to {session_id}: {data.get('type')}")
        except Exception as e:
            logger.warning(f"WebSocket send error for session {session_id}: {e}. Removing client.")
            websockets.pop(session_id, None) # Remove broken connection

# --- Agent Callback Functions ---
async def done_callback(history: Optional[AgentHistoryList], session_id: str): # history can be None on error
    """Callback executed when the agent's run() method finishes or errors."""
    if session_id not in agent_status and session_id not in agents: # Check both
        logger.warning(f"done_callback called for non-existent/cleaned-up session {session_id}")
        return

    try:
        success = history.is_successful() if history else False
        final_result = history.final_result() if history else "Agent run finished with an error."
        status = "done_success" if success else "done_failure"

        if not history: # If run failed with an exception before producing history
             status = "error"
             final_result = "Agent task failed unexpectedly."

        agent_status[session_id] = status # Update final status

        logger.info(f"Agent {session_id} finished. Status: {status}. Success: {success}. Result: {final_result}")

        # --- Send final update ---
        # Optionally, you could loop through history.history here and send
        # intermediate steps if needed, but sending just the final result is simpler now.
        final_screenshot = None
        if history and history.history:
            final_screenshot = history.history[-1].state.screenshot

        await send_update(session_id, {
            "type": "done",
            "status": status,
            "success": success,
            "final_result": final_result
        })
        # Optionally send the final screenshot if available
        if final_screenshot:
             await send_update(session_id, {
                 "type": "screenshot",
                 "data": final_screenshot
             })

    except Exception as e:
         logger.error(f"Error in done_callback for session {session_id}: {e}", exc_info=True)
         agent_status[session_id] = "error"
         await send_update(session_id, {
             "type": "error",
             "status": "error",
             "message": f"Error processing agent completion: {e}"
         })
    finally:
        # Clean up agent resources
        logger.info(f"Cleaning up resources for session {session_id}")
        agents.pop(session_id, None) # Remove agent instance
        agent_tasks.pop(session_id, None) # Remove task reference
        # Let agent_status reflect the final outcome until a new task starts
        # websockets.pop(session_id, None) # Keep WS open briefly to ensure 'done' message sends


# --- API Endpoints ---
@app.post("/agent/task", response_model=AgentResponse)
async def run_agent_task(request: TaskRequest):
    """Endpoint to start a new browser agent task."""
    session_id = request.session_id or str(uuid.uuid4())

    # Check if an agent task is already running for this session
    if session_id in agent_tasks and not agent_tasks[session_id].done():
         raise HTTPException(status_code=409, detail=f"Agent is already running a task for session {session_id}.")

    logger.info(f"Received task for session {session_id}: '{request.task}'")

    try:
        # Get the configured LLM instance
        llm = get_llm_instance()
        logger.info(f"Using LLM: {llm.__class__.__name__}")

        # Prepare Agent Configuration
        agent = Agent(
            task=request.task,
            llm=llm,
            browser=shared_browser,
            generate_gif=False,
        )

        # Store agent instance and initial status
        agents[session_id] = agent
        agent_status[session_id] = "starting"

        # --- Define wrapped done_callback ONLY ---
        async def wrapped_done_callback(history: Optional[AgentHistoryList]): # Accept Optional history
             if session_id in agent_status: # Check status dict as agent might be popped
                 await done_callback(history, session_id)
             else:
                  logger.warning(f"Done callback skipped for cleaned-up session {session_id}")

        # Run the agent's task in the background using asyncio.create_task
        logger.info(f"Starting agent.run for session {session_id}")
        # --- REMOVE on_step_end ---
        agent_run_task = asyncio.create_task(
             agent.run(max_steps=50) # No step callback passed
        )

        # --- Modify task completion handler slightly ---
        def task_completion_handler(future: asyncio.Task):
             history_result: Optional[AgentHistoryList] = None
             if future.cancelled():
                  logger.info(f"Agent task for session {session_id} was cancelled.")
                  agent_status[session_id] = "error"
                  asyncio.create_task(send_update(session_id, {"type": "status", "status": "cancelled", "message": "Task cancelled"}))
                  agents.pop(session_id, None)
                  agent_tasks.pop(session_id, None)
                  return # Don't call done_callback on cancellation

             elif exception := future.exception():
                  logger.error(f"Agent task for session {session_id} failed: {exception}", exc_info=exception)
                  # history_result remains None
             else:
                  # Task completed successfully, get history
                  history_result = future.result() # This should be AgentHistoryList

             # Call done_callback regardless of success/failure, passing history (or None)
             asyncio.create_task(wrapped_done_callback(history_result))

        agent_run_task.add_done_callback(task_completion_handler)

        # Store the task and update status
        agent_tasks[session_id] = agent_run_task
        agent_status[session_id] = "running" # Status update after scheduling

        logger.info(f"Agent task successfully scheduled for session {session_id}")
        return AgentResponse(session_id=session_id, status="running", message="Agent task started.")

    except HTTPException as http_exc:
         # Re-raise specific HTTP exceptions (e.g., from LLM config)
         logger.error(f"HTTP Exception during task start for {session_id}: {http_exc.detail}")
         agent_status[session_id] = "error"
         raise http_exc
    except Exception as e:
        logger.error(f"Generic Exception during task start for {session_id}: {e}", exc_info=True)
        agent_status[session_id] = "error"
        detail = f"Failed to start agent task: {e}"
        raise HTTPException(status_code=500, detail=detail)

@app.get("/agent/{session_id}/status")
async def get_agent_status(session_id: str):
    """Endpoint to get the current status of an agent session."""
    if session_id not in agent_status and session_id not in agents: # Check both
        raise HTTPException(status_code=404, detail="Agent session not found")

    status = agent_status.get(session_id, "unknown")
    goal = ""
    last_action = ""
    step = 0

    # Try to get more details if the agent instance still exists
    if agent := agents.get(session_id):
        try:
            history_list = agent.state.history.history
            if history_list:
                last_item = history_list[-1]
                step = agent.state.n_steps
                if last_item.model_output:
                    goal = last_item.model_output.current_state.next_goal
                    if last_item.model_output.action:
                        try:
                            first_action = last_item.model_output.action[0]
                            action_dump = first_action.model_dump(exclude_unset=True)
                            action_name = list(action_dump.keys())[0]
                            params_str = str(action_dump[action_name])
                            last_action = f"{action_name}({params_str[:100]}{'...' if len(params_str)>100 else ''})"
                        except Exception:
                            last_action = "Error formatting"
        except Exception as e:
            logger.warning(f"Could not retrieve detailed status for running agent {session_id}: {e}")
            goal = "Error fetching state"
            last_action = "Error fetching state"

    return {
        "session_id": session_id,
        "status": status,
        "current_goal": goal,
        "last_action": last_action,
        "step": step
    }

# --- WebSocket Endpoint ---
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handles WebSocket connections for real-time agent updates."""
    await websocket.accept()
    websockets[session_id] = websocket
    logger.info(f"WebSocket connected for session {session_id}")
    try:
        # Send initial status if agent already has one
        initial_status = agent_status.get(session_id, "idle") # Default to idle if unknown
        logger.info(f"Sending initial WS status for {session_id}: {initial_status}")
        await send_update(session_id, {"type": "status", "status": initial_status, "goal": "Connecting..."})

        # Keep the connection alive, mostly for sending server->client updates
        while True:
            # We don't expect many messages from client in this setup,
            # but can add handling here if needed (e.g., pause/resume commands)
            # try:
            #     data = await websocket.receive_text()
            #     logger.debug(f"WS Received from {session_id}: {data}")
            #     # Handle client messages if necessary
            # except WebSocketDisconnect:
            #     # This will be caught by the outer handler
            #     break

            # Periodic ping to keep connection alive (optional, depends on infra)
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}", exc_info=True)
    finally:
        # Clean up WebSocket connection
        logger.info(f"Cleaning up WebSocket for session {session_id}")
        websockets.pop(session_id, None)
        # Optionally try to close again, but it might already be closed
        try:
            await websocket.close()
        except Exception:
            pass


# --- Application Startup and Shutdown ---
@app.on_event("startup")
async def startup_event():
    """Actions to perform on server startup."""
    logger.info("FastAPI application starting up...")
    # Perform initial LLM configuration check
    try:
        get_llm_instance()
        logger.info("Initial LLM configuration check successful.")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed initial LLM configuration check: {e}. Fix config and restart.")
        # Depending on deployment, you might want to exit or prevent startup
        # For now, we'll log critically and let it continue, but it will fail on first request
    # Initialize the shared browser (optional, can be lazy-loaded too)
    # await shared_browser._init() # Pre-start the browser if desired
    logger.info("Startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on server shutdown."""
    logger.info("FastAPI application shutting down...")
    # Cancel any running agent tasks gracefully
    active_tasks = list(agent_tasks.values()) # Copy values before iterating
    if active_tasks:
        logger.info(f"Cancelling {len(active_tasks)} running agent tasks...")
        for task in active_tasks:
            if not task.done():
                task.cancel()
        # Allow some time for tasks to acknowledge cancellation
        await asyncio.gather(*active_tasks, return_exceptions=True)
        logger.info("Agent tasks cancelled.")

    # Close WebSocket connections
    active_websockets = list(websockets.values()) # Copy values
    if active_websockets:
        logger.info(f"Closing {len(active_websockets)} WebSocket connections...")
        for ws in active_websockets:
            try:
                await ws.close(code=1001) # Going away
            except Exception:
                pass # Ignore errors on close
        logger.info("WebSocket connections closed.")

    # Close the shared browser instance
    logger.info("Closing shared browser...")
    await shared_browser.close()
    logger.info("Shared browser closed.")

    logger.info("Shutdown complete.")

# --- Main Execution Guard ---
if __name__ == "__main__":
    # Uvicorn setup for running directly (python main.py)
    # Reload should generally be False for production, True for dev via command line
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)), # Use PORT env var if available
        log_level=log_level_str.lower(), # Use configured log level
        reload=False # Uvicorn's reload is better handled via CLI flag (--reload)
    )
