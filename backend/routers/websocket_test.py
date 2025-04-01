# backend/routers/websocket_test.py

import asyncio
import base64
import logging
import time
from typing import Dict, Any

# --- Import Playwright ---
from playwright.async_api import async_playwright, Playwright, Browser, Page, Error as PlaywrightError

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

# Get a logger specific to this module
logger = logging.getLogger(__name__)

# Create a router instance specifically for these test routes
router = APIRouter()


@router.websocket("/ws/test")
async def websocket_test_endpoint(websocket: WebSocket):
    """
    Dedicated endpoint for testing WebSocket connection and basic browser automation
    using a DIRECT Playwright instance (not browser-use's shared one).
    """
    await websocket.accept()
    logger.info("Direct Playwright WS Test: Connection accepted.")

    playwright_instance: Playwright | None = None
    browser: Browser | None = None
    page: Page | None = None

    async def send_test_update(data: Dict[str, Any]):
        """Helper to send updates for the test connection."""
        try:
            if websocket.client_state == WebSocketDisconnect:
                raise WebSocketDisconnect  # Ensure we don't try sending to closed socket
            await websocket.send_json(data)
        except WebSocketDisconnect:
            logger.info(
                "Direct Playwright WS Test: Client disconnected during send.")
            raise  # Re-raise to exit the main loop
        except Exception as e:
            logger.error(
                f"Direct Playwright WS Test: Error sending update: {e}")
            # Optional: attempt to send an error message back to client
            # try: await websocket.send_json({"type": "error", "message": "Failed to send update"})
            # except: pass

    try:
        # 1. Launch Playwright and Browser
        await send_test_update({
            "type": "status", "status": "starting", "step": 1, "goal": "Launch Browser",
            "last_action": "Launching Playwright...", "summary_text": "Starting test: Launching browser.",
            "timestamp": time.time()
        })
        logger.info("Direct Playwright WS Test: Launching Playwright")
        playwright_instance = await async_playwright().start()
        logger.info(
            "Direct Playwright WS Test: Launching Chromium (headful for test)")
        # Launch headful for easier visual debugging during test
        # Launch headful
        browser = await playwright_instance.chromium.launch(headless=False)
        logger.info("Direct Playwright WS Test: Creating new page")
        page = await browser.new_page()
        await asyncio.sleep(1)

        # 2. Go to Google
        await send_test_update({
            "type": "status", "status": "running", "step": 2, "goal": "Navigate to Google",
            "last_action": "page.goto(https://www.google.com)", "summary_text": "Navigating to google.com.",
            "timestamp": time.time()
        })
        logger.info("Direct Playwright WS Test: Navigating to Google")
        await page.goto("https://www.google.com", wait_until="domcontentloaded")
        await asyncio.sleep(1)

        # 3. Send Screenshot after Navigation
        logger.info(
            "Direct Playwright WS Test: Taking screenshot after navigation.")
        screenshot_bytes = await page.screenshot(type="png", full_page=False)
        screenshot_b64_str = base64.b64encode(screenshot_bytes).decode('utf-8')
        logger.info(
            f"Direct Playwright WS Test: Sending nav screenshot (length: {len(screenshot_b64_str)}).")
        await send_test_update({"type": "screenshot", "data": screenshot_b64_str})
        await asyncio.sleep(0.5)

        # 4. Type "apple"
        search_box_selector = 'textarea[name="q"]'
        await send_test_update({
            "type": "status", "status": "running", "step": 3, "goal": "Search for 'apple'",
            "last_action": f"page.locator('{search_box_selector}').fill('apple')",
            "summary_text": "Typing 'apple' into the search bar.", "timestamp": time.time()
        })
        logger.info("Direct Playwright WS Test: Typing 'apple'.")
        await page.locator(search_box_selector).fill("apple")
        await asyncio.sleep(1)

        # 5. Send Screenshot after Typing
        logger.info(
            "Direct Playwright WS Test: Taking screenshot after typing.")
        screenshot_bytes = await page.screenshot(type="png", full_page=False)
        screenshot_b64_str = base64.b64encode(screenshot_bytes).decode('utf-8')
        logger.info(
            f"Direct Playwright WS Test: Sending typing screenshot (length: {len(screenshot_b64_str)}).")
        await send_test_update({"type": "screenshot", "data": screenshot_b64_str})
        await asyncio.sleep(1)

        # 6. Done
        await send_test_update({
            "type": "done", "status": "done_success", "success": True,
            "final_result": "Direct Playwright test sequence completed.",
            "summary_text": "Direct Playwright test finished successfully.",
            "timestamp": time.time()
        })
        logger.info("Direct Playwright WS Test: Sequence finished.")

    except WebSocketDisconnect:
        logger.info("Direct Playwright WS Test: Client disconnected.")
    except PlaywrightError as e:  # Catch specific Playwright errors
        logger.error(
            f"Direct Playwright WS Test: Playwright Error: {e}", exc_info=True)
        try:
            await send_test_update({
                "type": "error", "status": "error",
                "message": f"Playwright Error during test: {e}",
                "summary_text": "A browser control error occurred during the test sequence.",
                "timestamp": time.time()})
        except Exception:
            pass  # Ignore if sending fails on error
    except Exception as e:
        logger.error(
            f"Direct Playwright WS Test: General Error: {e}", exc_info=True)
        try:
            await send_test_update({
                "type": "error", "status": "error",
                "message": f"An error occurred during the test: {e}",
                "summary_text": "An error occurred during the test sequence.",
                "timestamp": time.time()})
        except Exception:
            pass  # Ignore if sending fails on error
    finally:
        # --- Cleanup Playwright resources specific to this test connection ---
        logger.info("Direct Playwright WS Test: Cleaning up resources...")
        if page:
            try:
                await page.close()
                logger.info("Direct Playwright WS Test: Page closed.")
            except Exception as e:
                logger.warning(
                    f"Direct Playwright WS Test: Error closing page: {e}")
        if browser:
            try:
                await browser.close()
                logger.info("Direct Playwright WS Test: Browser closed.")
            except Exception as e:
                logger.warning(
                    f"Direct Playwright WS Test: Error closing browser: {e}")
        if playwright_instance:
            try:
                await playwright_instance.stop()
                logger.info("Direct Playwright WS Test: Playwright stopped.")
            except Exception as e:
                logger.warning(
                    f"Direct Playwright WS Test: Error stopping Playwright: {e}")

        # Close WebSocket
        try:
            if websocket.client_state != WebSocketDisconnect:
                await websocket.close()
        except Exception:
            pass
        logger.info("Direct Playwright WS Test: Connection closed.")
