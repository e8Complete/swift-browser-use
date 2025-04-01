import asyncio
import json
import logging
import os
import uuid
import pytest
import websockets
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
API_URL = os.getenv("TEST_API_URL", "http://localhost:8000")
WS_URL = os.getenv("TEST_WS_URL", "ws://localhost:8000/ws")


class TestAgentSummaries:
    @pytest.fixture
    def session_id(self):
        """Generate a consistent session ID for each test"""
        return str(uuid.uuid4())

    @pytest.fixture
    async def websocket(self, session_id):
        """Fixture to create and cleanup a WebSocket connection"""
        ws = None
        try:
            ws = await websockets.connect(f"{WS_URL}/{session_id}")
            yield ws
        finally:
            if ws:
                await ws.close()

    @pytest.fixture
    def api_session(self):
        """Fixture to create a requests session"""
        session = requests.Session()
        yield session
        session.close()

    async def create_agent_task(self, session, task, session_id):
        """Helper to create an agent task"""
        response = session.post(
            f"{API_URL}/agent/task",
            json={"task": task, "session_id": session_id}
        )
        assert response.status_code == 202
        return response.json()["session_id"]

    async def collect_messages(self, websocket, timeout=30):
        """Collect WebSocket messages until done or timeout"""
        messages = []
        try:
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                data = json.loads(message)
                messages.append(data)
                if data["type"] in ["done", "error"]:
                    break
        except asyncio.TimeoutError:
            logger.warning("Message collection timed out")
        return messages

    @pytest.mark.asyncio
    async def test_status_summaries(self, websocket, api_session, session_id):
        """Test that status updates include proper summaries"""
        # Create a task
        await self.create_agent_task(
            api_session,
            "Go to google.com and search for 'python testing'",
            session_id
        )

        # Collect messages
        messages = await self.collect_messages(websocket)

        # Check for summaries
        status_messages = [m for m in messages if m["type"] == "status"]
        summaries = [m.get("summary_text")
                     for m in status_messages if m.get("summary_text")]

        assert len(summaries) > 0, "No summaries received"

        # Check summary format
        for summary in summaries:
            assert "I need to" in summary, f"Invalid summary format: {summary}"
            assert "I am now" in summary, f"Invalid summary format: {summary}"

    @pytest.mark.asyncio
    async def test_action_translations(self, websocket, api_session, session_id):
        """Test that different actions are translated correctly"""
        await self.create_agent_task(
            api_session,
            "Go to google.com, type 'python', and click search",
            session_id
        )

        messages = await self.collect_messages(websocket)
        summaries = [m.get("summary_text")
                     for m in messages if m.get("summary_text")]

        # Check for different action phrases
        action_phrases = [
            "navigating to",
            "typing",
            "clicking"
        ]

        found_phrases = set()
        for summary in summaries:
            for phrase in action_phrases:
                if phrase in summary:
                    found_phrases.add(phrase)

        assert len(found_phrases) > 0, "No action phrases found in summaries"

    @pytest.mark.asyncio
    async def test_completion_summaries(self, websocket, api_session, session_id):
        """Test completion/done message summaries"""
        await self.create_agent_task(
            api_session,
            "Go to example.com",
            session_id
        )

        messages = await self.collect_messages(websocket)
        done_message = next((m for m in messages if m["type"] == "done"), None)

        assert done_message is not None, "No done message received"
        assert "summary_text" in done_message, "Done message missing summary"

        summary = done_message["summary_text"]
        if done_message["success"]:
            assert "finished the task" in summary
        else:
            assert "encountered an issue" in summary or "something went wrong" in summary

    @pytest.mark.asyncio
    async def test_error_summaries(self, websocket, api_session, session_id):
        """Test error message summaries"""
        # Create a task likely to fail
        await self.create_agent_task(
            api_session,
            "Go to nonexistent-website-12345.com",
            session_id
        )

        messages = await self.collect_messages(websocket)
        error_message = next(
            (m for m in messages if m["type"] == "error"), None)

        if error_message:
            assert "summary_text" in error_message, "Error message missing summary"
            assert "sorry" in error_message["summary_text"].lower(
            ), "Error summary should be apologetic"

    @pytest.mark.asyncio
    async def test_screenshot_updates(self, websocket, api_session, session_id):
        """Test that screenshots are sent without excessive logging"""
        await self.create_agent_task(
            api_session,
            "Go to google.com",
            session_id
        )

        messages = await self.collect_messages(websocket)
        screenshot_messages = [
            m for m in messages if m["type"] == "screenshot"]

        assert len(screenshot_messages) > 0, "No screenshots received"
        for msg in screenshot_messages:
            assert "data" in msg, "Screenshot message missing data"


def main():
    """Manual test runner"""
    import asyncio
    import sys

    async def run_tests():
        # Create test instance
        test = TestAgentSummaries()
        session_id = str(uuid.uuid4())

        # Create fixtures
        async with websockets.connect(f"{WS_URL}/{session_id}") as ws:
            with requests.Session() as session:
                # Run tests
                await test.test_status_summaries(ws, session, session_id)
                await test.test_action_translations(ws, session, session_id)
                await test.test_completion_summaries(ws, session, session_id)
                await test.test_error_summaries(ws, session, session_id)
                await test.test_screenshot_updates(ws, session, session_id)

        print("✅ All tests completed successfully!")
        return True

    try:
        asyncio.run(run_tests())
        sys.exit(0)
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
