# backend/main.py

import asyncio
import base64
import json  # For serializing model actions
import logging
import os
import uuid
from typing import Dict, Optional, Any, List

import uvicorn
from browser_use import Agent, Browser, BrowserConfig, AgentHistoryList
# Import necessary views
from browser_use.agent.views import AgentState, BrowserStateHistory
# Use browser-use logger setup
from browser_use.utils import logger as browser_use_logger
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status, APIRouter, Header
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketState  # Add WebSocketState import
import time  # For timestamps

# --- Load Environment Variables ---
# Ensure this runs before other imports that might depend on env vars
load_dotenv()

# --- Logging Setup (Leverage browser-use logger) ---
log_level_str = os.getenv("BROWSER_USE_LOGGING_LEVEL", "INFO").upper()
log_format = '%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
# Use browser-use's setup, but ensure our logger uses it too
browser_use_logger.setLevel(getattr(logging, log_level_str, logging.INFO))
formatter = logging.Formatter(log_format)
# Make sure there's a handler attached (e.g., StreamHandler)
if not browser_use_logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    browser_use_logger.addHandler(ch)
# Get our own logger instance which will inherit browser-use's level/handlers
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
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not set")
            if not model_name:
                raise ValueError("AGENT_LLM_MODEL not set for openai")
            return ChatOpenAI(model=model_name, temperature=0)

        elif provider == "azure":
            from langchain_azure_openai import AzureChatOpenAI
            if not os.getenv("AZURE_ENDPOINT"):
                raise ValueError("AZURE_ENDPOINT not set")
            if not os.getenv("AZURE_OPENAI_API_KEY"):
                raise ValueError("AZURE_OPENAI_API_KEY not set")
            if not model_name:
                raise ValueError(
                    "AGENT_LLM_MODEL (Azure deployment name) not set")
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
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY not set")
            if not model_name:
                model_name = "claude-3-sonnet-20240229"
            return ChatAnthropic(model=model_name, temperature=0)

        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            if not os.getenv("GEMINI_API_KEY"):
                raise ValueError("GEMINI_API_KEY not set")
            if not model_name:
                model_name = "gemini-1.5-flash-latest"
            return ChatGoogleGenerativeAI(model=model_name, temperature=0, convert_system_message_to_human=True)

        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            if not model_name:
                raise ValueError("AGENT_LLM_MODEL not set for ollama")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            logger.info(
                f"Connecting to Ollama at {base_url} with model {model_name}")
            return ChatOllama(model=model_name, base_url=base_url, temperature=0)

        else:
            raise ValueError(f"Unsupported AGENT_LLM_PROVIDER: {provider}")

    except ImportError as e:
        logger.error(
            f"Missing LangChain package for provider '{provider}': {e}. Please install it (e.g., `uv pip install langchain-{provider}` or similar).")
        raise HTTPException(
            status_code=501, detail=f"LLM provider '{provider}' integration not installed.")
    except ValueError as e:
        logger.error(f"Configuration error for LLM provider '{provider}': {e}")
        raise HTTPException(
            status_code=500, detail=f"LLM configuration error: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize LLM: {e}")


# --- FastAPI App Setup ---
app = FastAPI(title="Swift Agent Backend")

# --- CORS Configuration ---
# Stricter origins are better for production
origins_str = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
origins = [origin.strip()
           for origin in origins_str.split(",") if origin.strip()]
logger.info(f"Configuring CORS allowed origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Agent Management State ---
agents: Dict[str, Agent] = {}
# e.g., "idle", "running", "done_success", "done_failure", "error", "cancelled"
agent_status: Dict[str, str] = {}
websockets: Dict[str, WebSocket] = {}
agent_tasks: Dict[str, asyncio.Task] = {}

# --- Shared Browser Instance ---
# Configure Browser based on environment if needed
# e.g., BROWSER_HEADLESS=true, BROWSER_DISABLE_SECURITY=true etc.
browser_headless = os.getenv(
    "BROWSER_HEADLESS", "False").lower() in ('true', '1', 't')
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
    # Check state before sending (CORRECT)
    if ws and ws.client_state == WebSocketState.CONNECTED:
        try:
            await ws.send_json(data)
        except WebSocketDisconnect:
            logger.warning(
                f"WebSocket disconnected during send for session {session_id}. Removing client.")
            websockets.pop(session_id, None)
        except Exception as e:
            logger.warning(
                f"WebSocket send error for session {session_id}: {e}. Removing client.")
            websockets.pop(session_id, None)
    # else: logger.debug(f"WebSocket not connected for session {session_id}, skipping send.")

# --- Agent Step Simulation Helper (Replaces direct hook) ---
# This task runs alongside the agent to periodically check state and send updates.


async def agent_monitor_task(session_id: str, agent: Agent):
    logger.info(f"Starting monitor task for session {session_id}")
    last_sent_step = -1
    last_sent_goal = ""
    last_sent_action = ""
    last_screenshot_timestamp = 0
    screenshot_interval = 1.0  # Seconds between sending screenshots

    while session_id in agents and agent_status.get(session_id) == "running":
        await asyncio.sleep(0.75)  # Check slightly more frequently

        if session_id not in agents:
            break  # Agent was removed

        try:
            current_step = agent.state.n_steps
            current_goal = "Waiting for next step..."
            current_action = "Processing..."
            screenshot_b64 = None
            current_time = time.time()

            # Get latest info from history if available
            if agent.state.history and agent.state.history.history:
                last_history_item = agent.state.history.history[-1]
                if last_history_item.model_output:
                    current_goal = last_history_item.model_output.current_state.next_goal or "Analyzing..."
                    if last_history_item.model_output.action:
                        try:
                            # Handle potential list of actions
                            action_strs = []
                            for action_model in last_history_item.model_output.action:
                                action_dump = action_model.model_dump(
                                    exclude_unset=True)
                                action_name = list(action_dump.keys())[0]
                                action_params = action_dump[action_name]
                                params_str = str(action_params)
                                if len(params_str) > 100:
                                    params_str = params_str[:100] + "..."
                                action_strs.append(
                                    f"{action_name}({params_str})")
                            current_action = "; ".join(
                                action_strs) if action_strs else "No action specified"
                        except Exception:
                            current_action = "Error formatting action"
                    else:
                        current_action = "Thinking..."  # No action yet from LLM

                # Get screenshot if available and enough time has passed
                if isinstance(last_history_item.state, BrowserStateHistory) and last_history_item.state.screenshot:
                    if current_time - last_screenshot_timestamp > screenshot_interval:
                        screenshot_b64 = last_history_item.state.screenshot
                        last_screenshot_timestamp = current_time

            # Send status update only if something changed
            if current_step != last_sent_step or current_goal != last_sent_goal or current_action != last_sent_action:
                await send_update(session_id, {
                    "type": "status",
                    "status": "running",
                    "step": current_step,
                    "goal": current_goal,
                    "last_action": current_action,
                    "timestamp": current_time
                })
                last_sent_step = current_step
                last_sent_goal = current_goal
                last_sent_action = current_action

            # Send screenshot if available
            if screenshot_b64:
                # logger.info(f"Sending screenshot for {session_id}") # Maybe too noisy
                await send_update(session_id, {
                    "type": "screenshot",
                    "data": screenshot_b64,
                    "timestamp": current_time
                })
            # else: logger.debug(f"No new screenshot available for {session_id}") # Debug level

        except Exception as e:
            logger.error(
                f"Error in agent_monitor_task for {session_id}: {e}", exc_info=True)
            # Optional: Send an error update via WebSocket
            await send_update(session_id, {"type": "error", "message": f"Monitor task error: {e}"})
            await asyncio.sleep(5)  # Avoid tight loop on error

    logger.info(f"Stopping monitor task for session {session_id}")

# --- Agent Done Callback (Modified for Debug Info) ---


async def agent_done_callback(session_id: str, history: Optional[AgentHistoryList]):
    """Called when agent.run() finishes or errors."""
    if session_id not in agent_status and session_id not in agents:  # Check both dictionaries
        logger.warning(
            f"agent_done_callback called for cleaned-up session {session_id}")
        return

    logger.info(
        f"Agent run finished for session {session_id}. Processing result...")
    status = "error"  # Default status
    success = False
    final_result = "Agent run finished unexpectedly."
    debug_info_payload = {}

    try:
        if history:
            success = history.is_successful()
            final_result = history.final_result() or (
                "Task completed successfully." if success else "Task completed with issues.")
            status = "done_success" if success else "done_failure"

            # --- Extract Debug Info ---
            try:
                # Safely get model actions, default to empty list if None
                raw_model_actions = history.model_actions() or []
                debug_info_payload = {
                    "urls": history.urls() or [],
                    "action_names": history.action_names() or [],
                    "extracted_content": history.extracted_content() or [],
                    # Convert errors to strings
                    "errors": [str(e) for e in (history.errors() or [])],
                    # FIX: Convert action representations to string safely
                    "model_actions": [str(a) for a in raw_model_actions]
                }
                logger.info(f"Extracted debug info for {session_id}: "
                            f"URLs={len(debug_info_payload['urls'])}, "
                            f"Actions={len(debug_info_payload['action_names'])}, "
                            f"Content={len(debug_info_payload['extracted_content'])}, "
                            f"Errors={len(debug_info_payload['errors'])}, "
                            f"ModelActions={len(debug_info_payload['model_actions'])}")
            except Exception as debug_err:
                logger.error(
                    f"Failed to extract debug info for {session_id}: {debug_err}", exc_info=True)
                # Optionally send minimal debug info or none at all
                debug_info_payload = {"errors": [
                    f"Debug info extraction failed: {debug_err}"]}

        else:
            # No history object likely means an internal agent error before completion
            status = "error"
            final_result = "Agent task failed unexpectedly (no history returned)."

        # Update final status before sending messages
        agent_status[session_id] = status

        logger.info(
            f"Final status for {session_id}: {status}. Success: {success}. Result: '{final_result}'")

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
                logger.warning(
                    f"Could not get final screenshot for {session_id}: {e}")

        if final_screenshot:
            await send_update(session_id, {
                "type": "screenshot",
                "data": final_screenshot,
                "timestamp": time.time()
            })

         # --- Send Debug Info ---
        if debug_info_payload:
            logger.info(f"Sending debug_info payload for {session_id}")
            debug_info_payload["type"] = "debug_info"  # Add message type
            await send_update(session_id, debug_info_payload)

    except Exception as e:
        logger.error(
            f"Error in agent_done_callback for {session_id}: {e}", exc_info=True)
        agent_status[session_id] = "error"  # Ensure status reflects error
        await send_update(session_id, {
            "type": "error",
            "status": "error",
            "message": f"Error processing agent completion: {e}",
            "timestamp": time.time()
        })
    finally:
        # Agent instance is NOT removed here to allow conversation
        # Need separate logic for timeout/cleanup later
        agent_tasks.pop(session_id, None)  # Task is finished
        logger.info(
            f"Agent task removed for session {session_id}. Agent instance kept for potential follow-up.")


# --- Task Completion Handler Helper ---
def create_task_completion_handler(session_id: str, agent: Agent):
    """Creates the handler function for agent task completion."""
    async def wrapped_done_callback(history: Optional[AgentHistoryList]):
        # Check if the session still exists before processing
        if session_id in agents or session_id in agent_status:
            await agent_done_callback(session_id, history)
        else:
            logger.warning(
                f"wrapped_done_callback skipped for already cleaned-up session {session_id}")

    def task_completion_handler(future: asyncio.Task):
        history_result: Optional[AgentHistoryList] = None
        if future.cancelled():
            logger.info(f"Agent task for session {session_id} was cancelled.")
            # Check if session still relevant before updating status/sending messages
            if session_id in agents or session_id in agent_status:
                agent_status[session_id] = "cancelled"
                asyncio.create_task(send_update(session_id, {
                                    "type": "status", "status": "cancelled", "message": "Task cancelled by server."}))
                # Clean up cancelled task resources
                agents.pop(session_id, None)
                agent_tasks.pop(session_id, None)
                # Also close WS for cancelled task
                websockets.pop(session_id, None)
        elif exception := future.exception():
            logger.error(
                f"Agent task for session {session_id} failed: {exception}", exc_info=exception)
            # Don't call done_callback on exception, let it handle the error state if needed
            # Ensure status is marked as error if not already
            if session_id in agent_status and agent_status[session_id] != 'error':
                agent_status[session_id] = "error"
                asyncio.create_task(send_update(session_id, {
                                    "type": "error", "status": "error", "message": f"Agent task execution failed: {exception}"}))
            # The done_callback might still be called separately if agent.run catches internally
            # but we ensure error state is set here.
        else:  # Handle success
            history_result = future.result()
            # Call done callback only on successful completion or graceful failure (history returned)
            asyncio.create_task(wrapped_done_callback(history_result))

    return task_completion_handler

# --- API Endpoints ---


@app.post("/agent/task", response_model=AgentResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_or_update_agent_task(request: TaskRequest):
    """
    Starts a new agent task or sends a message to an existing idle agent.
    Returns 202 Accepted immediately, updates happen via WebSocket.
    """
    session_id = request.session_id or str(uuid.uuid4())
    task_description = request.task

    try:
        llm_instance = get_llm_instance()
        logger.info(
            f"Using LLM for session {session_id}: {llm_instance.__class__.__name__}")
    except Exception as e:
        logger.error(f"LLM Initialization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"LLM configuration error: {e}")

    # Check if agent exists and is idle (ready for new input)
    if session_id in agents and agent_status.get(session_id) not in ["running", "starting"]:
        agent = agents[session_id]
        logger.info(
            f"Adding message to existing agent {session_id}: '{task_description}'")
        try:
            # Inject message into agent's history
            agent.message_manager._add_message_with_tokens(
                HumanMessage(content=task_description))
            # Restart the agent's run loop (or trigger next step if possible)
            # For now, we restart the run task if it's not already running
            if session_id not in agent_tasks or agent_tasks[session_id].done():
                # Set status before starting task
                agent_status[session_id] = "running"
                await send_update(session_id, {"type": "status", "status": "running", "goal": "Processing new input..."})

                # --- Define wrapped done_callback ---
                async def wrapped_done_callback(history: Optional[AgentHistoryList]):
                    if session_id in agent_status:
                        await agent_done_callback(session_id, history)

                logger.info(f"Restarting agent.run for session {session_id}")
                agent_run_task = asyncio.create_task(agent.run(max_steps=50))

                def task_completion_handler(future: asyncio.Task):
                    history_result: Optional[AgentHistoryList] = None
                    session_id_from_task = future.get_name()

                    if future.cancelled():
                        logger.info(
                            f"Agent task for session {session_id_from_task} was cancelled.")
                        if session_id_from_task in agent_status:
                            agent_status[session_id_from_task] = "error"
                        asyncio.create_task(send_update(session_id_from_task, {
                            "type": "status",
                            "status": "cancelled",
                            "message": "Task cancelled"
                        }))
                        agents.pop(session_id_from_task, None)
                        agent_tasks.pop(session_id_from_task, None)
                    elif exception := future.exception():
                        logger.error(
                            f"Agent task failed: {exception}", exc_info=exception)
                        asyncio.create_task(wrapped_done_callback(None))
                    else:
                        history_result = future.result()
                        asyncio.create_task(
                            wrapped_done_callback(history_result))

                agent_run_task.add_done_callback(task_completion_handler)
                agent_tasks[session_id] = agent_run_task
                # Start monitor task
                asyncio.create_task(agent_monitor_task(
                    session_id, agent))

                return AgentResponse(session_id=session_id, status="running", message="Agent task restarted with new input.")
            else:
                # Agent exists but task is somehow still marked as running - unusual state
                logger.warning(
                    f"Agent {session_id} exists but task is still active? Sending message only.")
                # Inform user
                await send_update(session_id, {"type": "system", "message": "New input added."})
                return AgentResponse(session_id=session_id, status="running", message="Input added to running agent.")

        except Exception as e:
            logger.error(
                f"Error adding message to agent {session_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to add message to agent: {e}")

    # Check if agent is currently busy
    elif session_id in agent_tasks and not agent_tasks[session_id].done():
        logger.warning(
            f"Task rejected for session {session_id}: Agent already running.")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"Agent is already running a task for session {session_id}.")

    # Start a new agent session
    else:
        logger.info(
            f"Starting new agent session {session_id} with task: '{task_description}'")
        try:
            logger.info(f"Using LLM: {llm_instance.__class__.__name__}")

            agent = Agent(
                task=task_description,
                llm=llm_instance,
                browser=shared_browser,
                generate_gif=False,
            )
            agents[session_id] = agent
            # Set status before starting task
            agent_status[session_id] = "running"

            await send_update(session_id, {"type": "status", "status": "running", "goal": "Initializing..."})

            # --- Define wrapped done_callback ---
            async def wrapped_done_callback(history: Optional[AgentHistoryList]):
                if session_id in agent_status:
                    await agent_done_callback(session_id, history)

            logger.info(f"Starting agent.run for new session {session_id}")
            agent_run_task = asyncio.create_task(
                agent.run(max_steps=50))  # No step callback here

            def task_completion_handler(future: asyncio.Task):
                history_result: Optional[AgentHistoryList] = None
                session_id_from_task = future.get_name()

                if future.cancelled():
                    logger.info(
                        f"Agent task for session {session_id_from_task} was cancelled.")
                    if session_id_from_task in agent_status:
                        agent_status[session_id_from_task] = "error"
                    asyncio.create_task(send_update(session_id_from_task, {
                        "type": "status",
                        "status": "cancelled",
                        "message": "Task cancelled"
                    }))
                    agents.pop(session_id_from_task, None)
                    agent_tasks.pop(session_id_from_task, None)
                elif exception := future.exception():
                    logger.error(
                        f"Agent task failed: {exception}", exc_info=exception)
                    asyncio.create_task(wrapped_done_callback(None))
                else:
                    history_result = future.result()
                    asyncio.create_task(wrapped_done_callback(history_result))

            agent_run_task.add_done_callback(task_completion_handler)
            agent_tasks[session_id] = agent_run_task
            # Start monitor task
            asyncio.create_task(agent_monitor_task(
                session_id, agent))

            return AgentResponse(session_id=session_id, status="running", message="New agent task started.")

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to start agent task: {e}")


@app.get("/agent/{session_id}/status")
async def get_agent_status(session_id: str):
    """Endpoint to get the current status of an agent session."""
    if session_id not in agent_status and session_id not in agents:  # Check both
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
                            action_dump = first_action.model_dump(
                                exclude_unset=True)
                            action_name = list(action_dump.keys())[0]
                            params_str = str(action_dump[action_name])
                            last_action = f"{action_name}({params_str[:100]}{'...' if len(params_str)>100 else ''})"
                        except Exception:
                            last_action = "Error formatting"
        except Exception as e:
            logger.warning(
                f"Could not retrieve detailed status for running agent {session_id}: {e}")
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
    # Allow all origins from env var for flexibility, or default to localhost
    allowed_origins_str = os.getenv(
        "WEBSOCKET_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    allowed_origins = [o.strip()
                       for o in allowed_origins_str.split(",") if o.strip()]
    # '*' allows any origin
    is_wildcard_allowed = "*" in allowed_origins
    # Check if the specific origin header is present and in the allowed list OR wildcard is set
    is_origin_allowed = (
        is_wildcard_allowed or
        (origin and origin in allowed_origins)
    )
    # Allow connections without an Origin header (e.g., simple ws clients like websocat) if wildcard is not set OR if origin list is empty?
    # Let's be explicit: Allow if wildcard OR origin matches. If no origin header, only allow if wildcard is set.
    if not origin and not is_wildcard_allowed:
        logger.warning(
            f"WebSocket connection rejected for {session_id}: Missing Origin header and wildcard not allowed.")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    logger.info(
        f"WebSocket attempting connection for {session_id}. Origin: '{origin}'. Allowed: {allowed_origins}. Decision: {'Allow' if is_origin_allowed else 'Deny'}")

    if not is_origin_allowed:
        logger.warning(
            f"WebSocket connection rejected for {session_id} due to invalid origin: '{origin}'")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # --- Connection Accepted ---
    try:
        await websocket.accept()
        logger.info(f"WebSocket ACCEPTED for session {session_id}")

        # Store the WebSocket connection
        websockets[session_id] = websocket

        # Send initial status based on current state
        initial_status = agent_status.get(
            session_id, "idle")  # Default to idle if unknown
        if session_id not in agents and initial_status == "idle":
            logger.info(
                f"New WebSocket connection for idle session {session_id}.")
            await send_update(session_id, {"type": "status", "status": "idle", "message": "Connected, ready for task."})
        elif session_id in agents:
            current_status = agent_status.get(session_id, "unknown")
            logger.info(
                f"WebSocket reconnected for existing session {session_id} with status {current_status}")
            # Send current full status immediately on reconnect
            # Reuse status endpoint logic
            status_details = await get_agent_status(session_id)
            await send_update(session_id, {**status_details, "type": "status", "message": "Reconnected."})
            # Restart monitor task if agent is running but monitor might have stopped
            if current_status == "running" and (session_id not in agent_tasks or agent_tasks[session_id].done()):
                logger.info(
                    f"Restarting monitor task for reconnected running session {session_id}")
                asyncio.create_task(agent_monitor_task(
                    session_id, agents[session_id]))
        else:
            # Should ideally not happen if initial_status wasn't idle
            logger.warning(
                f"WebSocket connected for unknown session {session_id} with non-idle status {initial_status}. Resetting to idle.")
            agent_status[session_id] = "idle"
            await send_update(session_id, {"type": "status", "status": "idle", "message": "Connected, state unclear, ready for task."})

        # --- Message Handling Loop ---
        while True:
            try:
                data = await websocket.receive_json()
                logger.debug(f"WS Received from {session_id}: {data}")

                # Handle chat messages to inject into the agent
                if data.get("type") == "chat_message":
                    content = data.get("content")
                    if content and session_id in agents:
                        agent = agents[session_id]
                        current_status = agent_status.get(session_id)
                        task_is_running = session_id in agent_tasks and not agent_tasks[session_id].done(
                        )

                        if not task_is_running:  # Only inject and restart if not actively running
                            logger.info(
                                f"Injecting user message into agent {session_id} and restarting task: '{content}'")
                            agent.message_manager._add_message_with_tokens(
                                HumanMessage(content=content))
                            # Restart the task
                            agent_status[session_id] = "running"
                            await send_update(session_id, {"type": "status", "status": "running", "goal": "Processing new input..."})
                            logger.info(
                                f"Restarting agent.run via WS for session {session_id}")
                            agent_run_task = asyncio.create_task(
                                agent.run(max_steps=50))
                            handler = create_task_completion_handler(
                                session_id, agent)
                            agent_run_task.add_done_callback(handler)
                            agent_tasks[session_id] = agent_run_task
                            asyncio.create_task(
                                agent_monitor_task(session_id, agent))
                            await send_update(session_id, {"type": "system", "message": "Processing your message..."})
                        else:
                            logger.warning(
                                f"WS message received for {session_id}, but task is currently active. Message ignored or queued (if implemented).")
                            await send_update(session_id, {"type": "system", "message": "Agent is busy, please wait for the current action to complete."})

                    elif not content:
                        await send_update(session_id, {"type": "error", "message": "Received empty chat message."})
                    else:  # session_id not in agents
                        await send_update(session_id, {"type": "error", "message": "Agent session not found or inactive."})

                # Handle other potential message types from client (e.g., ping)
                elif data.get("type") == "ping":
                    await send_update(session_id, {"type": "pong"})

            except WebSocketDisconnect:
                logger.info(
                    f"WebSocket disconnected for session {session_id} (client closed).")
                break
            except ValidationError as e:  # Handle Pydantic validation errors if using models for receiving
                logger.warning(
                    f"Invalid WS message format from {session_id}: {e}")
                await send_update(session_id, {"type": "error", "message": f"Invalid message format: {e}"})
            except Exception as e:
                logger.error(
                    f"Error processing WS message from {session_id}: {e}", exc_info=True)
                try:
                    # Try sending an error back before breaking
                    await send_update(session_id, {"type": "error", "message": f"Error processing message: {str(e)}"})
                except Exception:
                    pass  # Ignore send error if connection is already broken
                break  # Exit loop on significant error

    except Exception as e:
        logger.error(
            f"Error during WebSocket connection setup for {session_id}: {e}", exc_info=True)
    finally:
        logger.info(
            f"Cleaning up WebSocket connection for session {session_id}")
        websockets.pop(session_id, None)
        # Ensure WS is closed if loop exited unexpectedly
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:  # Compare state to state
                await websocket.close(code=1011)  # Internal Error
        except Exception:
            pass

# Include the WebSocket router in the main app
app.include_router(ws_router)

# --- Application Startup and Shutdown ---


@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up...")
    try:
        get_llm_instance()
        logger.info("Initial LLM configuration check successful.")
    except Exception as e:
        logger.critical(
            f"CRITICAL: Failed initial LLM configuration check: {e}. Fix config and restart.")
    # await shared_browser._init() # Optional: Pre-start browser
    logger.info("Startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down...")
    shutdown_timeout = 10.0  # Seconds to wait for cleanup steps

    # --- Close shared browser FIRST (often the most problematic) ---
    logger.info("Attempting to close shared browser...")
    try:
        # Wait for close but with a timeout
        await asyncio.wait_for(shared_browser.close(), timeout=shutdown_timeout / 2)
        logger.info("Shared browser closed successfully.")
    except asyncio.TimeoutError:
        logger.warning(
            f"Timeout closing shared browser after {shutdown_timeout / 2}s. It might be left running.")
    except Exception as e:
        # Catch errors like "Connection closed" if browser died already
        # Keep log concise
        logger.error(f"Error closing shared browser: {e}", exc_info=False)

    # --- Cancel running agent tasks ---
    active_task_ids = list(agent_tasks.keys())
    if active_task_ids:
        logger.info(
            f"Cancelling {len(active_task_ids)} running agent tasks...")
        tasks_to_await = []
        for session_id in active_task_ids:
            task = agent_tasks.pop(session_id, None)
            if task and not task.done():
                task.cancel()
                # Collect tasks to await cancellation
                tasks_to_await.append(task)

        if tasks_to_await:
            logger.info(
                f"Waiting up to {shutdown_timeout / 2}s for tasks to cancel...")
            # Wait for tasks to finish cancellation, but with a timeout
            done, pending = await asyncio.wait(tasks_to_await, timeout=shutdown_timeout / 2, return_when=asyncio.ALL_COMPLETED)

            if pending:
                logger.warning(
                    f"{len(pending)} agent tasks did not cancel within the timeout.")
            # Log results/exceptions from completed/cancelled tasks
            for task in done:
                try:
                    task.result()  # Access result to raise exceptions if any occurred during cancellation handling
                except asyncio.CancelledError:
                    # logger.info(f"Task {task.get_name()} cancelled successfully.") # Can be noisy
                    pass
                except Exception as task_exc:
                    logger.warning(
                        f"Exception during task cancellation processing for task {task.get_name()}: {task_exc}")

        logger.info("Agent tasks cancellation process finished.")
    else:
        logger.info("No active agent tasks to cancel.")

    # --- Close active WebSocket connections ---
    active_ws_ids = list(websockets.keys())
    if active_ws_ids:
        logger.info(f"Closing {len(active_ws_ids)} WebSocket connections...")
        ws_close_tasks = []
        for session_id in active_ws_ids:
            ws = websockets.pop(session_id, None)
            if ws and ws.client_state == status.WS_STATE_CONNECTED:
                # Close gracefully
                ws_close_tasks.append(ws.close(code=1001))  # 1001 = Going Away
        if ws_close_tasks:
            # Give websockets a short time to close
            await asyncio.wait(ws_close_tasks, timeout=2.0)
        logger.info("WebSocket connections closed.")
    else:
        logger.info("No active WebSocket connections to close.")

    # Clear remaining state dictionaries (belt-and-braces)
    agents.clear()
    agent_status.clear()
    websockets.clear()
    agent_tasks.clear()

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
        reload=False  # Use --reload flag via CLI for development
    )
