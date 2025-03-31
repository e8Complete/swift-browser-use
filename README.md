# Swift Browser Agent UI

A voice-controlled web interface for interacting with a browser automation agent. This project combines advanced voice recognition with browser automation to create a hands-free way to control web browsing.

## 🚀 Features

- **Voice Input**: Uses [@ricky0123/vad-react](https://github.com/ricky0123/vad-react) for browser-based Voice Activity Detection (VAD)
- **Transcription**: Configurable to use different Speech-to-Text services (Groq Whisper, OpenAI Whisper API)
- **Agent Backend**: Communicates with Python backend API in `/backend` directory
- **Browser Control**: Uses the `browser-use` library to automate browser actions based on voice commands
- **Real-time Updates**: WebSockets for live agent status, goals, actions, and browser screenshots
- **Modern Stack**: Built with Next.js (React) and Tailwind CSS, deployable on Vercel or similar platforms

## 📋 Project Structure

This repository contains both the frontend UI and the Python backend API in a monorepo structure:

```
└── swift-browser-ui/ (Repo root)
    ├── app/              <-- Next.js frontend code
    ├── backend/          <-- Python FastAPI backend code
    ├── public/           <-- Frontend public assets (VAD files)
    ├── node_modules/     <-- Node.js dependencies
    ├── backend/.venv/    <-- Python virtual environment (created by uv/venv)
    ├── package.json      <-- Node.js dependencies
    ├── backend/pyproject.toml <-- Python project config
    ├── backend/requirements.txt <-- Python dependencies (for pip)
    ├── .env.local        <-- Frontend & Shared Env Vars (Root)
    ├── backend/.env      <-- Backend-specific Env Vars
    └── ... (other config files)
```

## 🏗️ Architecture

The system works through these components:

1. **Frontend (Browser)**:

   - Captures voice using VAD to detect speech
   - Sends audio blob to `/api/route.ts`
   - Displays agent progress via WebSocket updates

2. **Next.js API Route**:

   - Receives audio and sends to configured STT service
   - Gets transcription and forwards to Python backend

3. **Python Backend**:
   - Manages browser-use Agent instance using configured LLM
   - Executes browser actions based on commands
   - Sends status updates and screenshots to frontend

```
+-----------------------+  POST /api  +-----------------------+  POST /agent/task  +---------------------+  +-------------+
| Frontend (Browser)    |------------->| Next.js API Route     |-------------------->| Python Backend     |-->| browser-use |
| - VAD (Detects Speech)|  (Audio Blob)| - STT Transcription   |    (Task Text)     | - FastAPI/WebSocket|   | Agent       |
| - WebSocket Client    |              | - Calls Python Backend|                    | - Manages Agent    |   |             |
| - Displays Agent View |<-------------+                       |<--------------------+                     |<--+             |
+-----------------------+ WebSocket JSON+-----------------------+  JSON Response     +---------------------+  +-------------+
                         (Status, Screenshot)                     (Task Accepted + SessionID)
```

## 📋 Prerequisites

- **Node.js**: v18 or later recommended
- **pnpm**: `npm install -g pnpm` for frontend dependencies
- **Python**: v3.11 or later
- **uv** (Recommended) or pip: For Python dependencies
  - Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`
- **API Keys**:
  - STT provider (Groq or OpenAI) for frontend
  - LLM provider (OpenAI, Anthropic, Google, Azure) for backend
- **Ollama** (Optional): For local models
- **Microphone**: A working microphone for your browser

## 🔧 Setup & Development

### 1. Clone Your Fork

```bash
# Replace with your fork's URL
git clone https://github.com/e8Complete/swift-browser-use.git
cd swift-browser-use
```

### 2. Install Frontend Dependencies

From the root directory:

```bash
pnpm install
pnpm add openai # Required if using OpenAI for STT
```

### 3. Install Backend Dependencies

Navigate to the backend directory:

```bash
cd backend
```

#### Option A - Recommended with uv

```bash
# This creates/uses .venv and installs from pyproject.toml
uv pip install -e .
```

#### Option B - Using pip with requirements.txt

```bash
# Generate requirements.txt if needed
uv pip freeze > requirements.txt

# Create a virtual environment and install
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Install Playwright Browser

After activating the virtual environment:

```bash
playwright install chromium
```

### 5. Return to Root

```bash
cd ..
```

### 6. Set Up Environment Variables

#### Frontend (.env.local at the root)

Copy `.env.example` to `.env.local`: `cp .env.example .env.local`

Edit `.env.local`:

```env
# ./ai-ng-swift/.env.local

# --- STT Configuration for Next.js API Route ---
# Choose one: 'groq' or 'openai'
STT_PROVIDER=openai

# Required if STT_PROVIDER=groq
GROQ_API_KEY=gr_...

# Required if STT_PROVIDER=openai
OPENAI_API_KEY=sk_...

# --- Backend URLs ---
# URL for Next.js API route to call Python backend
PYTHON_BACKEND_URL=http://localhost:8000

# WebSocket URL for the browser client (JS) to connect
# Use ws:// locally, wss:// when deployed with SSL
# Needs NEXT_PUBLIC_ prefix!
NEXT_PUBLIC_PYTHON_WS_URL=ws://localhost:8000
```

#### Backend (backend/.env)

Create this file inside the backend/ directory:

```env
# ./ai-ng-swift/backend/.env

# --- LLM Configuration for Python Agent ---
# Choose provider: 'openai', 'azure', 'anthropic', 'google', 'ollama'
AGENT_LLM_PROVIDER=ollama
# Model name appropriate for the provider
AGENT_LLM_MODEL=llama3

# --- Provider Specific Settings (only need those for selected provider) ---
# OPENAI_API_KEY=sk_... # If using openai LLM
# ANTHROPIC_API_KEY=...
# GEMINI_API_KEY=...
# AZURE_ENDPOINT=...
# AZURE_OPENAI_API_KEY=...
# AZURE_OPENAI_API_VERSION=...
OLLAMA_BASE_URL=http://localhost:11434 # Default if using ollama

# --- ElevenLabs Configuration ---
# Get your API key from https://elevenlabs.io/
ELEVENLABS_API_KEY=your_api_key_here
# Optional: Set a specific voice ID (default will be used if not specified)
ELEVENLABS_VOICE_ID=your_voice_id_here
# Optional: Set model (defaults to 'eleven_monolingual_v1')
ELEVENLABS_MODEL_ID=eleven_monolingual_v1

# Optional backend logging level
# BROWSER_USE_LOGGING_LEVEL=debug
```

### 7. Run the Development Servers

#### Terminal 1 (Frontend)

From the root directory:

```bash
pnpm dev
```

Access at http://localhost:3000

#### Terminal 2 (Backend)

From the backend/ directory:

```bash
cd backend
# If using pip/venv, activate it: source .venv/bin/activate
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API runs at http://localhost:8000

### 8. Test

1. Open http://localhost:3000
2. Grant microphone permissions
3. Speak a command (e.g., "Open Google and search for cute cat videos")
4. Watch the browser UI for transcription, status updates, and screenshots

### 8. WebSocket Test Route

The backend includes a test WebSocket route for development and debugging purposes. You can connect to it at:

```
ws://localhost:8000/ws/test
```

This route accepts the following message types:

- `ping`: Server responds with `pong`
- `echo`: Server echoes back the message
- `status`: Server sends a mock status update

Example using browser console:

```javascript
// Connect to test WebSocket
const ws = new WebSocket("ws://localhost:8000/ws/test");
```

## 🔄 Recent Updates

### WebSocket Improvements

- Enhanced WebSocket state handling using `WebSocketState` from Starlette
- Improved connection state checks for more reliable message delivery
- Better cleanup of disconnected WebSocket connections
- Proper state comparison for WebSocket lifecycle management

### Debug Information

- Added comprehensive debug info extraction from agent runs
- Debug payload includes:
  - URLs visited
  - Action names executed
  - Extracted content
  - Errors encountered
  - Model actions taken
- Improved error handling and logging throughout the application

### Error Handling

- Enhanced error handling in WebSocket communication
- Better management of connection states and disconnections
- Improved logging for debugging connection issues
- Graceful handling of connection cleanup

### Environment Variables

The following new environment variables are available for configuration:

```env
# WebSocket allowed origins (comma-separated)
WEBSOCKET_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Browser configuration
BROWSER_HEADLESS=false  # Set to true for headless mode
```

## 🐛 Debugging

### WebSocket Connection Issues

If you encounter WebSocket connection problems:

1. Check the allowed origins in your environment configuration
2. Verify the WebSocket URL matches your deployment setup
3. Monitor the backend logs for connection state changes
4. Ensure proper cleanup of connections in development

### Debug Information

To access debug information during agent runs:

1. Monitor the WebSocket messages for type "debug_info"
2. Check the backend logs for detailed state extraction
3. Review the agent's history for comprehensive debugging
4. Use the status endpoint for current agent state

## 🚀 Deployment

Deploying a monorepo with mixed languages requires separate configurations:

### Frontend (Root /)

Deploy the Next.js app to Vercel, Netlify, etc:

- Configure platform to use Node.js with pnpm
- Set environment variables in the platform settings:
  - `STT_PROVIDER`
  - `GROQ_API_KEY` or `OPENAI_API_KEY`
  - `PYTHON_BACKEND_URL` (must point to deployed backend)
  - `NEXT_PUBLIC_PYTHON_WS_URL` (must point to deployed backend)

### Backend (/backend)

Deploy to a platform suitable for long-running Python processes:

- Fly.io, Render, Cloud Run, Railway, DigitalOcean Apps, AWS EC2/ECS
- Often deployed via Docker

#### Dockerfile Example (place in backend/)

```dockerfile
# Use an official Python base image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by Playwright/Browsers
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Playwright dependencies
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libexpat1 libgbm1 libgcc1 libglib2.0-0 \
    libpango-1.0-0 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 \
    libxdamage1 libxext6 libxfixes3 libxrandr2 libxtst6 \
    ca-certificates fonts-liberation libappindicator3-1 \
    libasound2 libatspi2.0-0 libcairo2 libfontconfig1 \
    libgtk-3-0 libpangoft2-1.0-0 libstdc++6 \
    lsb-release wget xdg-utils \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy only dependency definition files first for caching
COPY pyproject.toml ./
# Optional: If using requirements.txt
# COPY requirements.txt ./

# Install Python dependencies
RUN uv pip install --system --no-cache -e .
# Or if using requirements.txt:
# RUN uv pip install --system --no-cache -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium --with-deps

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📜 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

- [browser-use](https://github.com/user/browser-use) for browser automation
- [@ricky0123/vad-react](https://github.com/ricky0123/vad-react) for voice activity detection
- [Next.js](https://nextjs.org/) and [FastAPI](https://fastapi.tiangolo.com/) frameworks
