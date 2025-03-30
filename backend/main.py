# backend/main.py

import asyncio
import base64
import logging
import os
import uuid
from typing import Dict, Optional, Any, List

import uvicorn
from browser_use import Agent, Browser, BrowserConfig, AgentHistoryList
from browser_use.agent.views import AgentState, BrowserStateHistory # Import necessary views
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status, APIRouter, Header
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage # Import for context injection
from pydantic import BaseModel
import time # For timestamps

# --- Load Environment Variables ---
# Ensure this runs before other imports that might depend on env vars
load_dotenv()

# --- Basic Logging Setup ---
# Configure logging level (consider using BROWSER_USE_LOGGING_LEVEL if set)
log_level_str = os.getenv("BROWSER_USE_LOGGING_LEVEL", "INFO").upper()
log_format = '%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
logging.basicConfig(level=getattr(logging, log_level_str, logging.INFO), format=log_format)
# Suppress overly verbose logs from libraries if needed
# logging.getLogger("websockets.server").setLevel(logging.WARNING)
# logging.getLogger("websockets.protocol").setLevel(logging.WARNING)
# logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
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
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
logger.info(f"Configuring CORS allowed origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Agent Management State ---
# Using simple dictionaries for state. Consider Redis/DB for production scalability.
agents: Dict[str, Agent] = {}
agent_status: Dict[str, str] = {} # e.g., "idle", "running", "awaiting_input", "done_success", "error"
websockets: Dict[str, WebSocket] = {}
agent_tasks: Dict[str, asyncio.Task] = {} # Stores the asyncio task running the agent

# --- Shared Browser Instance ---
# Configure Browser based on environment if needed
# e.g., BROWSER_HEADLESS=true, BROWSER_DISABLE_SECURITY=true etc.
browser_headless = os.getenv("BROWSER_HEADLESS", "False").lower() in ('true', '1', 't')
# Add more browser config vars as needed
shared_browser = Browser(config=BrowserConfig(headless=browser_headless))
logger.info(f"Browser configured (headless={browser_headless})")

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
        except Exception as e:
            logger.warning(f"WebSocket send error for session {session_id}: {e}. Removing client.")
            websockets.pop(session_id, None)

# --- Agent Step Simulation Helper (Replaces direct hook) ---
# This task runs alongside the agent to periodically check state and send updates.
async def agent_monitor_task(session_id: str, agent: Agent):
    logger.info(f"Starting monitor task for session {session_id}")
    last_sent_step = -1
    last_sent_goal = ""
    last_sent_action = ""

    while session_id in agents and agent_status.get(session_id) == "running":
        await asyncio.sleep(1.5) # Check every 1.5 seconds

        if session_id not in agents: break # Agent was removed

        try:
            current_step = agent.state.n_steps
            current_goal = "Waiting for next step..."
            current_action = "Processing..."
            screenshot_b64 = None

            # Get latest info from history if available
            if agent.state.history and agent.state.history.history:
                last_history_item = agent.state.history.history[-1]
                if last_history_item.model_output:
                     current_goal = last_history_item.model_output.current_state.next_goal
                     if last_history_item.model_output.action:
                         try:
                             first_action_model = last_history_item.model_output.action[0]
                             action_dump = first_action_model.model_dump(exclude_unset=True)
                             action_name = list(action_dump.keys())[0]
                             action_params = action_dump[action_name]
                             params_str = str(action_params)
                             if len(params_str) > 100: params_str = params_str[:100] + "..."
                             current_action = f"{action_name}({params_str})"
                         except Exception: current_action = "Error formatting action"

                if isinstance(last_history_item.state, BrowserStateHistory) and last_history_item.state.screenshot:
                    screenshot_b64 = last_history_item.state.screenshot

            # Send status update only if something changed
            if current_step != last_sent_step or current_goal != last_sent_goal or current_action != last_sent_action:
                 await send_update(session_id, {
                     "type": "status",
                     "status": "running",
                     "step": current_step,
                     "goal": current_goal,
                     "last_action": current_action,
                     "timestamp": time.time()
                 })
                 last_sent_step = current_step
                 last_sent_goal = current_goal
                 last_sent_action = current_action

            # Always send screenshot if available in the latest history step
            if screenshot_b64:
                logger.info(f"Sending screenshot for {session_id}")
                await send_update(session_id, {
                    "type": "screenshot",
                    "data": screenshot_b64
                })
            else:
                logger.info(f"No screenshot available for {session_id}")

        except Exception as e:
            logger.error(f"Error in agent_monitor_task for {session_id}: {e}")
            # Optional: Send an error update via WebSocket
            await asyncio.sleep(5) # Avoid tight loop on error

    logger.info(f"Stopping monitor task for session {session_id}")

# --- Agent Done Callback ---
async def agent_done_callback(session_id: str, history: Optional[AgentHistoryList]):
    """Called when agent.run() finishes or errors."""
    if session_id not in agent_status: # Check status dict, as agent might be popped already
        logger.warning(f"agent_done_callback called for cleaned-up session {session_id}")
        return

    logger.info(f"Agent run finished for session {session_id}. Processing result...")
    try:
        success = history.is_successful() if history else False
        final_result = history.final_result() if history else "Agent run finished with an error or no history."
        status = "done_success" if success else "done_failure"

        if not history: # Explicitly mark as error if history is None
             status = "error"
             final_result = "Agent task failed unexpectedly (no history)."

        agent_status[session_id] = status # Update final status

        logger.info(f"Final status for {session_id}: {status}. Success: {success}. Result: '{final_result}'")

        # Send final 'done' message
        await send_update(session_id, {
            "type": "done",
            "status": status,
            "success": success,
            "final_result": final_result,
            "timestamp": time.time()
        })

        # Send the very last screenshot if available
        final_screenshot = None
        if history and history.history:
            try:
                 # Ensure state is BrowserStateHistory before accessing screenshot
                 if isinstance(history.history[-1].state, BrowserStateHistory):
                     final_screenshot = history.history[-1].state.screenshot
            except Exception as e:
                 logger.warning(f"Could not get final screenshot for {session_id}: {e}")

        if final_screenshot:
             await send_update(session_id, {
                 "type": "screenshot",
                 "data": final_screenshot
             })

    except Exception as e:
         logger.error(f"Error in agent_done_callback for session {session_id}: {e}", exc_info=True)
         agent_status[session_id] = "error"
         await send_update(session_id, {
             "type": "error",
             "status": "error",
             "message": f"Error processing agent completion: {e}",
             "timestamp": time.time()
         })
    finally:
        # Agent instance is NOT removed here to allow conversation
        # Need separate logic for timeout/cleanup later
        agent_tasks.pop(session_id, None) # Task is finished
        logger.info(f"Agent task removed for session {session_id}. Agent instance kept.")


# --- API Endpoints ---
@app.post("/agent/task", response_model=AgentResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_or_update_agent_task(request: TaskRequest):
    """
    Starts a new agent task or sends a message to an existing idle agent.
    Returns 202 Accepted immediately, updates happen via WebSocket.
    """
    session_id = request.session_id or str(uuid.uuid4())
    task_description = request.task

    # Check if agent exists and is idle (ready for new input)
    if session_id in agents and agent_status.get(session_id) not in ["running", "starting"]:
        agent = agents[session_id]
        logger.info(f"Adding message to existing agent {session_id}: '{task_description}'")
        try:
            # Inject message into agent's history
            agent.message_manager._add_message_with_tokens(HumanMessage(content=task_description))
            # Restart the agent's run loop (or trigger next step if possible)
            # For now, we restart the run task if it's not already running
            if session_id not in agent_tasks or agent_tasks[session_id].done():
                agent_status[session_id] = "running" # Set status before starting task
                await send_update(session_id, {"type": "status", "status": "running", "goal": "Processing new input..."})

                # --- Define wrapped done_callback ---
                async def wrapped_done_callback(history: Optional[AgentHistoryList]):
                     if session_id in agent_status: await agent_done_callback(session_id, history)

                logger.info(f"Restarting agent.run for session {session_id}")
                agent_run_task = asyncio.create_task(agent.run(max_steps=50))

                def task_completion_handler(future: asyncio.Task):
                    history_result: Optional[AgentHistoryList] = None
                    if future.cancelled(): # Handle cancellation
                         logger.info(f"Agent task for session {session_id} was cancelled.")
                         agent_status[session_id] = "error" # Or 'cancelled'
                         asyncio.create_task(send_update(session_id, {"type": "status", "status": "cancelled", "message": "Task cancelled"}))
                         agents.pop(session_id, None); agent_tasks.pop(session_id, None)
                    elif exception := future.exception(): # Handle errors
                         logger.error(f"Agent task for session {session_id} failed: {exception}", exc_info=exception)
                    else: # Handle success
                         history_result = future.result()
                    # Call done callback in all non-cancelled cases
                    if not future.cancelled():
                         asyncio.create_task(wrapped_done_callback(history_result))

                agent_run_task.add_done_callback(task_completion_handler)
                agent_tasks[session_id] = agent_run_task
                # Start monitor task
                asyncio.create_task(agent_monitor_task(session_id, agent))

                return AgentResponse(session_id=session_id, status="running", message="Agent task restarted with new input.")
            else:
                # Agent exists but task is somehow still marked as running - unusual state
                logger.warning(f"Agent {session_id} exists but task is still active? Sending message only.")
                await send_update(session_id, {"type": "system", "message": "New input added."}) # Inform user
                return AgentResponse(session_id=session_id, status="running", message="Input added to running agent.")


        except Exception as e:
            logger.error(f"Error adding message to agent {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to add message to agent: {e}")

    # Check if agent is currently busy
    elif session_id in agent_tasks and not agent_tasks[session_id].done():
         logger.warning(f"Task rejected for session {session_id}: Agent already running.")
         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Agent is already running a task for session {session_id}.")

    # Start a new agent session
    else:
        logger.info(f"Starting new agent session {session_id} with task: '{task_description}'")
        try:
            llm = get_llm_instance()
            logger.info(f"Using LLM: {llm.__class__.__name__}")

            agent = Agent(
                task=task_description,
                llm=llm,
                browser=shared_browser,
                generate_gif=False,
            )
            agents[session_id] = agent
            agent_status[session_id] = "running" # Set status before starting task

            await send_update(session_id, {"type": "status", "status": "running", "goal": "Initializing..."})

             # --- Define wrapped done_callback ---
            async def wrapped_done_callback(history: Optional[AgentHistoryList]):
                 if session_id in agent_status: await agent_done_callback(session_id, history)


            logger.info(f"Starting agent.run for new session {session_id}")
            agent_run_task = asyncio.create_task(agent.run(max_steps=50)) # No step callback here

            def task_completion_handler(future: asyncio.Task):
                # (Same handler logic as above)
                history_result: Optional[AgentHistoryList] = None
                if future.cancelled(): # Handle cancellation
                    logger.info(f"Agent task for session {session_id} was cancelled.")
                    agent_status[session_id] = "error"
                    asyncio.create_task(send_update(session_id, {"type": "status", "status": "cancelled", "message": "Task cancelled"}))
                    agents.pop(session_id, None); agent_tasks.pop(session_id, None)
                elif exception := future.exception(): # Handle errors
                    logger.error(f"Agent task for session {session_id} failed: {exception}", exc_info=exception)
                else: # Handle success
                    history_result = future.result()
                if not future.cancelled():
                    asyncio.create_task(wrapped_done_callback(history_result))

            agent_run_task.add_done_callback(task_completion_handler)
            agent_tasks[session_id] = agent_run_task
             # Start monitor task
            asyncio.create_task(agent_monitor_task(session_id, agent))

            return AgentResponse(session_id=session_id, status="running", message="New agent task started.")

        except HTTPException as http_exc: raise http_exc
        except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to start agent task: {e}")

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

# --- WebSocket Router ---
ws_router = APIRouter()

@ws_router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    origin: Optional[str] = Header(None)
):
    # --- Explicit Origin Check ---
    allowed_origins = [o.strip() for o in os.getenv("WEBSOCKET_ALLOWED_ORIGINS", "http://localhost:3000").split(",")]
    # Use a flag for wildcard or allow if origin header is missing (e.g., from websocat)
    is_origin_allowed = (
        "*" in allowed_origins or
        origin is None or # Allow if no origin header (like websocat)
        origin in allowed_origins
    )

    logger.info(f"WebSocket attempting connection for {session_id}. Origin: '{origin}'. Allowed: {allowed_origins}. Decision: {'Allow' if is_origin_allowed else 'Deny'}")

    if not is_origin_allowed:
        logger.warning(f"WebSocket connection rejected for {session_id} due to invalid origin: '{origin}'")
        # Close before accepting if origin is invalid
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return # Important: Exit the function early

    try:
        await websocket.accept()
        logger.info(f"WebSocket ACCEPTED for session {session_id}")

        if session_id not in agent_status and session_id not in agents:
            logger.warning(f"WebSocket accepted for unknown session {session_id}.")
            agent_status[session_id] = "idle"
        websockets[session_id] = websocket
        initial_status = agent_status.get(session_id, "idle")
        await send_update(session_id, {"type": "status", "status": initial_status, "goal": "Connected."})
        if initial_status == "running" and session_id in agents and (session_id not in agent_tasks or agent_tasks[session_id].done()):
            logger.info(f"Restarting monitor task for reconnected session {session_id}")
            asyncio.create_task(agent_monitor_task(session_id, agents[session_id]))

        while True:
            try:
                data = await websocket.receive_json()
                logger.debug(f"WS Received from {session_id}: {data}")
                if data.get("type") == "chat_message":
                    content = data.get("content")
                    if content and session_id in agents:
                        agent = agents[session_id]
                        current_status = agent_status.get(session_id)
                        if current_status in ["running", "idle", "done_success", "done_failure", "error"]:
                            logger.info(f"Injecting user message into agent {session_id}: '{content}'")
                            agent.message_manager._add_message_with_tokens(HumanMessage(content=content))
                            if current_status != "running":
                                if session_id not in agent_tasks or agent_tasks[session_id].done():
                                    agent_status[session_id] = "running"
                                    await send_update(session_id, {"type": "status", "status": "running", "goal": "Processing new input..."})
                                    async def wrapped_done_callback(history: Optional[AgentHistoryList]):
                                        if session_id in agent_status: await agent_done_callback(session_id, history)
                                    logger.info(f"Restarting agent.run via WS for session {session_id}")
                                    agent_run_task = asyncio.create_task(agent.run(max_steps=50))
                                    def task_completion_handler(future: asyncio.Task):
                                        history_result: Optional[AgentHistoryList] = None
                                        if not future.cancelled():
                                            if exception := future.exception(): logger.error(f"Agent task {session_id} failed: {exception}", exc_info=exception)
                                            else: history_result = future.result()
                                            asyncio.create_task(wrapped_done_callback(history_result))
                                        else: logger.info(f"Agent task {session_id} was cancelled.")
                                    agent_run_task.add_done_callback(task_completion_handler)
                                    agent_tasks[session_id] = agent_run_task
                                    asyncio.create_task(agent_monitor_task(session_id, agent))
                                else: logger.warning(f"WS message for {session_id}, task unexpectedly active."); await send_update(session_id, {"type": "system", "message": "Input added."})
                            else: await send_update(session_id, {"type": "system", "message": "Input added."})
                        else: await send_update(session_id, {"type": "error", "message": f"Agent busy/not ready (status: {current_status})."})
                    else: await send_update(session_id, {"type": "error", "message": "Agent session not active."})
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
                break
            except Exception as e:
                logger.error(f"Error processing WS message from {session_id}: {e}", exc_info=True)
                try: await websocket.send_json({"type": "error", "message": f"Error processing message: {e}"})
                except Exception: break
    except Exception as e:
        logger.error(f"Error during WebSocket accept/handling for {session_id}: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up WebSocket connection for session {session_id}")
        websockets.pop(session_id, None)
        try:
            if websocket.client_state != WebSocketDisconnect:
                await websocket.close()
        except Exception: pass

# Include the WebSocket router in the main app
app.include_router(ws_router)

# --- Application Startup and Shutdown ---
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up...")
    try: get_llm_instance(); logger.info("Initial LLM configuration check successful.")
    except Exception as e: logger.critical(f"CRITICAL: Failed initial LLM configuration check: {e}. Fix config and restart.")
    # await shared_browser._init() # Optional: Pre-start browser
    logger.info("Startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down...")
    active_tasks = list(agent_tasks.values())
    if active_tasks:
        logger.info(f"Cancelling {len(active_tasks)} running agent tasks...")
        for task in active_tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*active_tasks, return_exceptions=True)
        logger.info("Agent tasks cancelled.")
    active_websockets = list(websockets.values())
    if active_websockets:
        logger.info(f"Closing {len(active_websockets)} WebSocket connections...")
        for ws in active_websockets:
            try: await ws.close(code=1001) # Going away
            except Exception: pass
        logger.info("WebSocket connections closed.")
    logger.info("Closing shared browser...")
    await shared_browser.close()
    logger.info("Shared browser closed.")
    logger.info("Shutdown complete.")

# --- Main Execution Guard ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting Uvicorn on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level_str.lower(),
        reload=False # Use --reload flag via CLI for development
    )
