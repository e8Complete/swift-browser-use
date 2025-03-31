// app/wstest/page.tsx
"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import clsx from "clsx";
import { LoadingIcon } from "@/lib/icons"; // Reuse loading icon if needed
import { toast } from "sonner"; // Added for error feedback

// Define the expected message structure (can reuse from main page if identical)
interface WsTestUpdateData {
  type: "status" | "screenshot" | "done" | "error";
  status?: string; // e.g., "starting", "running", "done_success", "error"
  step?: number;
  goal?: string;
  last_action?: string;
  summary_text?: string;
  data?: string; // For screenshot base64 data
  success?: boolean;
  final_result?: string;
  message?: string; // For errors
  timestamp?: number;
}

export default function WebSocketTestPage() {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isConnecting, setIsConnecting] = useState<boolean>(false); // Added connecting state
  const [agentStatus, setAgentStatus] = useState<string>("idle");
  const [lastMessage, setLastMessage] = useState<WsTestUpdateData | null>(null);
  const [screenshotData, setScreenshotData] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Function to connect (called by button)
  const connectWebSocket = useCallback(() => {
    // Prevent multiple connections if already connected or connecting
    if (wsRef.current || isConnecting) {
      console.log("WebSocket already connected or connecting.");
      toast.info("Already connected or attempting to connect.");
      return;
    }

    const wsBaseUrl = process.env.NEXT_PUBLIC_PYTHON_WS_URL;
    if (!wsBaseUrl) {
      console.error(
        "Error: NEXT_PUBLIC_PYTHON_WS_URL environment variable is not set."
      );
      setError("WebSocket URL configuration is missing.");
      setIsConnected(false);
      setIsConnecting(false); // Ensure connecting state is reset
      return;
    }
    const wsUrl = `${wsBaseUrl.replace(/\/$/, "")}/ws/test`;

    console.log(`Attempting to connect WebSocket to test endpoint: ${wsUrl}`);
    setError(null);
    setIsConnecting(true); // Set connecting state
    setAgentStatus("connecting..."); // Update status UI
    setLastMessage(null); // Clear last message
    setScreenshotData(null); // Clear screenshot

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket test connection opened.");
        setIsConnected(true);
        setIsConnecting(false); // Clear connecting state
        setAgentStatus("connected"); // Update status
        toast.success("WebSocket Connected!");
      };

      ws.onmessage = (event) => {
        // Reset connecting state if we receive a message (must be connected)
        if (isConnecting) setIsConnecting(false);
        try {
          const data: WsTestUpdateData = JSON.parse(event.data as string);
          console.log("Received WebSocket test message:", data);
          setLastMessage(data);

          switch (data.type) {
            case "status":
              setAgentStatus(data.status || "unknown");
              break;
            case "screenshot":
              setScreenshotData(data.data || null);
              break;
            case "done":
              setAgentStatus(data.status || "finished");
              if (data.data) setScreenshotData(data.data);
              // Don't auto-close here, let user disconnect or backend close
              break;
            case "error":
              setAgentStatus("error");
              setError(data.message || "Unknown backend error");
              // Attempt to close on error from backend
              disconnectWebSocket();
              break;
            default:
              console.warn("Unknown WebSocket test message type:", data.type);
          }
        } catch (parseError) {
          console.error("Error processing WebSocket test message:", parseError);
          setError("Received invalid message format from backend.");
          setAgentStatus("error");
          setIsConnecting(false);
          disconnectWebSocket(); // Close on parse error
        }
      };

      ws.onerror = (event) => {
        console.error("WebSocket test error:", event);
        setError(
          "WebSocket connection error occurred. Check backend logs and browser console."
        );
        setIsConnected(false);
        setIsConnecting(false);
        setAgentStatus("error");
        wsRef.current = null; // Ensure ref is cleared on error
        toast.error("WebSocket connection error.");
      };

      ws.onclose = (event) => {
        console.log(
          `WebSocket test connection closed. Code: ${event.code}, Reason: '${event.reason}'`
        );
        // Only update status if not already done/error
        if (
          agentStatus !== "done_success" &&
          agentStatus !== "error" &&
          agentStatus !== "finished"
        ) {
          setAgentStatus("disconnected");
        }
        setIsConnected(false);
        setIsConnecting(false);
        wsRef.current = null; // Ensure ref is cleared on close
        if (event.code !== 1000) {
          // 1000 is normal closure
          toast.warning(`WebSocket closed unexpectedly (Code: ${event.code})`);
        } else {
          toast.info("WebSocket disconnected.");
        }
      };
    } catch (connectionError: any) {
      console.error("Failed to create WebSocket for test:", connectionError);
      setError(
        `Failed to initiate WebSocket connection: ${connectionError.message}`
      );
      setIsConnected(false);
      setIsConnecting(false);
      setAgentStatus("error");
      toast.error("Failed to create WebSocket.");
    }
  }, [isConnecting, agentStatus]); // Add agentStatus here to reset UI on close if needed

  // Function to disconnect (called by button)
  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      console.log("Manually closing WebSocket test connection.");
      wsRef.current.close(1000, "Client disconnected"); // Use code 1000 for normal closure
      // ws.onclose handler will set state
    } else {
      toast.info("Already disconnected.");
    }
  }, []);

  // Effect just for cleanup on unmount
  useEffect(() => {
    // Return cleanup function
    return () => {
      if (wsRef.current) {
        console.log(
          "Closing WebSocket test connection due to component unmount."
        );
        // Prevent onclose handlers from updating state after unmount if possible
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
        wsRef.current.onmessage = null;
        wsRef.current.onopen = null;
        wsRef.current.close(1000, "Component unmounting");
        wsRef.current = null;
      }
    };
  }, []); // Empty dependency array ensures this runs only on mount/unmount

  return (
    <div className="flex flex-col items-center p-6 dark:text-white bg-white dark:bg-black min-h-screen">
      <h1 className="text-2xl font-bold mb-4">WebSocket Test Page</h1>

      {/* --- Control Buttons --- */}
      <div className="flex gap-4 mb-6">
        <button
          onClick={connectWebSocket}
          disabled={isConnected || isConnecting}
          className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isConnecting ? <LoadingIcon /> : "Connect"}
        </button>
        <button
          onClick={disconnectWebSocket}
          disabled={!isConnected && !isConnecting}
          className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Disconnect
        </button>
      </div>

      <div className="w-full max-w-4xl mb-6 p-4 border border-neutral-300 dark:border-neutral-700 rounded-lg bg-neutral-50 dark:bg-neutral-900 shadow-sm space-y-3">
        <p>
          <strong>Connection Status:</strong>{" "}
          <span
            className={clsx(
              "font-medium px-2 py-0.5 rounded",
              isConnecting
                ? "bg-yellow-200 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-100"
                : isConnected
                ? "bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-100"
                : "bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-100"
            )}
          >
            {isConnecting
              ? "Connecting..."
              : isConnected
              ? "Connected"
              : "Disconnected"}
          </span>
        </p>
        <p>
          <strong>Agent Status:</strong>{" "}
          <span className="font-mono text-sm bg-neutral-200 dark:bg-neutral-700 px-2 py-0.5 rounded">
            {agentStatus}
          </span>
        </p>
        {error && (
          <p className="text-red-500 dark:text-red-400">
            <strong>Error:</strong> {error}
          </p>
        )}
        <div className="pt-2 mt-2 border-t border-neutral-300 dark:border-neutral-700">
          <p className="text-xs font-semibold mb-1">Last Received Message:</p>
          <pre className="text-xs bg-neutral-100 dark:bg-black p-2 rounded overflow-x-auto max-h-60">
            {lastMessage
              ? JSON.stringify(lastMessage, null, 2)
              : "Waiting for messages..."}
          </pre>
        </div>
      </div>

      <h2 className="text-xl font-semibold mb-3">Live Screenshot</h2>
      <div className="w-full max-w-4xl border border-neutral-300 dark:border-neutral-700 rounded-md overflow-hidden bg-neutral-100 dark:bg-neutral-900 flex items-center justify-center aspect-[16/10]">
        {screenshotData ? (
          <img
            src={`data:image/png;base64,${screenshotData}`}
            alt="Live Agent Screenshot"
            className="object-contain max-w-full max-h-full"
            width={1280} // Example intrinsic size
            height={800} // Example intrinsic size
          />
        ) : (
          <span className="text-neutral-500 text-sm p-4">
            {agentStatus === "idle" || agentStatus === "disconnected"
              ? "Click Connect to start"
              : agentStatus === "connecting..."
              ? "Connecting..."
              : "Waiting for screenshot..."}
          </span>
        )}
      </div>
      <p className="mt-4 text-xs text-neutral-500 dark:text-neutral-400">
        Manually connect to the <code>/ws/test</code> backend endpoint using the
        button above.
      </p>
    </div>
  );
}
