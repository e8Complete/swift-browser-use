"use client";

import clsx from "clsx";
// Renamed useActionState to useFormState as per React 19 RC
import { useFormState } from "react-dom";
import { useEffect, useRef, useState, useCallback } from "react";
import { toast } from "sonner";
import {
  EnterIcon,
  LoadingIcon,
  MicrophoneIcon,
  StopCircleIcon,
} from "@/lib/icons";
// Removed usePlayer import
// import { usePlayer } from "@/lib/usePlayer";
import { track } from "@vercel/analytics";
import { useMicVAD, utils } from "@ricky0123/vad-react";

// Add a type for the error object to help TypeScript
interface VADError {
  message: string;
}

// --- Agent State Types ---
type AgentStatus =
  | "idle"
  | "starting"
  | "transcribing"
  | "running"
  | "paused"
  | "done_success"
  | "done_failure"
  | "error"
  | "disconnected"
  | "unknown";
interface AgentUpdateData {
  type: "status" | "screenshot" | "done" | "error" | "ping";
  status?: AgentStatus;
  step?: number;
  goal?: string;
  last_action?: string;
  data?: string; // For screenshot base64 data
  success?: boolean;
  final_result?: string;
  message?: string; // For errors
}

export default function Home() {
  const [input, setInput] = useState(""); // Holds text input content
  const [lastTranscript, setLastTranscript] = useState(""); // Holds the last spoken transcript
  const inputRef = useRef<HTMLInputElement>(null);
  // const player = usePlayer(); // Removed player

  // --- Agent State ---
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatus>("idle");
  const [currentGoal, setCurrentGoal] = useState<string>("");
  const [lastAction, setLastAction] = useState<string>("");
  const [screenshotData, setScreenshotData] = useState<string | null>(null);
  const [finalResult, setFinalResult] = useState<string | null>(null);
  const [agentStep, setAgentStep] = useState<number>(0);
  const wsRef = useRef<WebSocket | null>(null);

  // --- Form Submission State (using React 19's useFormState) ---
  const [formSubmissionState, formAction] = useFormState(handleSubmitAction, {
    status: "idle",
  });
  const isSubmitting = formSubmissionState.status === "submitting";

  const vad = useMicVAD({
    startOnLoad: false,
    onSpeechEnd: (audio) => {
      if (
        agentStatus !== "idle" &&
        agentStatus !== "error" &&
        agentStatus !== "done_success" &&
        agentStatus !== "done_failure"
      ) {
        console.log("Agent is busy, ignoring speech input from VAD.");
        return;
      }
      setAgentStatus("transcribing");
      const wav = utils.encodeWAV(audio);
      const blob = new Blob([wav], { type: "audio/wav" });
      const formData = new FormData();
      formData.append("input", blob, "audio.wav");
      if (sessionId) {
        formData.append("session_id", sessionId);
      }
      console.log("VAD detected speech end, submitting audio...");
      formAction(formData);
    },
    workletURL: "/vad.worklet.bundle.min.js",
    modelURL: "/silero_vad.onnx",
    positiveSpeechThreshold: 0.6,
    minSpeechFrames: 4,
    ortConfig(ort) {
      const isSafari = /^((?!chrome|android).)*safari/i.test(
        navigator.userAgent
      );

      ort.env.wasm = {
        wasmPaths: {
          "ort-wasm-simd-threaded.wasm": "/ort-wasm-simd-threaded.wasm",
          "ort-wasm-simd.wasm": "/ort-wasm-simd.wasm",
          "ort-wasm.wasm": "/ort-wasm.wasm",
          "ort-wasm-threaded.wasm": "/ort-wasm-threaded.wasm",
        },
        numThreads: isSafari ? 1 : 4,
      };
    },
  });

  const connectWebSocket = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) {
      console.log("Closing existing WebSocket connection before reconnecting.");
      wsRef.current.close();
      wsRef.current = null;
    }

    if (sessionId && !wsRef.current) {
      const wsUrlFromEnv = process.env.NEXT_PUBLIC_PYTHON_WS_URL;
      if (!wsUrlFromEnv) {
        console.error(
          "Error: NEXT_PUBLIC_PYTHON_WS_URL environment variable is not set."
        );
        toast.error("WebSocket configuration error.");
        setAgentStatus("error");
        return;
      }
      const wsUrl = `${wsUrlFromEnv.replace(/\/$/, "")}/${sessionId}`;

      console.log(`Connecting WebSocket to ${wsUrl}`);
      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log(`WebSocket connected for session ${sessionId}`);
          toast.info("Agent connection established.");
        };

        ws.onmessage = (event) => {
          try {
            const data: AgentUpdateData = JSON.parse(event.data as string);
            console.log("Received WebSocket message:", data);
            switch (data.type) {
              case "status":
                setAgentStatus(data.status || "unknown");
                if (data.goal) setCurrentGoal(data.goal);
                if (data.last_action) setLastAction(data.last_action);
                if (data.step) setAgentStep(data.step);
                break;
              case "screenshot":
                setScreenshotData(data.data || null);
                break;
              case "done":
                setAgentStatus(data.status || "unknown");
                setFinalResult(data.final_result || "Task finished.");
                setCurrentGoal("Task Complete");
                setLastAction("");
                setAgentStep(0);
                if (wsRef.current) {
                  wsRef.current.close();
                  wsRef.current = null;
                }
                break;
              case "ping":
                break;
              case "error":
                toast.error(`Agent Error: ${data.message || "Unknown error"}`);
                setAgentStatus("error");
                setCurrentGoal(`Error: ${data.message || "Unknown"}`);
                break;
              default:
                console.warn("Unknown WebSocket message type:", data.type);
            }
          } catch (error) {
            console.error("Error processing WebSocket message:", error);
            toast.error("Received invalid agent update.");
          }
        };

        ws.onerror = (error) => {
          console.error(`WebSocket error for session ${sessionId}:`, error);
          toast.error("Agent connection error.");
          setAgentStatus("error");
          setCurrentGoal("Connection error");
          wsRef.current = null;
        };

        ws.onclose = (event) => {
          console.log(
            `WebSocket disconnected for session ${sessionId}. Code: ${event.code}, Reason: '${event.reason}'`
          );
          if (
            agentStatus !== "done_success" &&
            agentStatus !== "done_failure" &&
            agentStatus !== "error"
          ) {
            setAgentStatus("disconnected");
          }
          wsRef.current = null;
        };
      } catch (error) {
        console.error("Failed to create WebSocket:", error);
        toast.error("Failed to initiate agent connection.");
        setAgentStatus("error");
        setCurrentGoal("Connection failed");
      }
    }
  }, [sessionId, agentStatus]);

  useEffect(() => {
    if (sessionId) {
      connectWebSocket();
    }
    return () => {
      if (wsRef.current) {
        console.log(
          "Closing WebSocket due to effect cleanup (unmount/sessionId change)."
        );
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [sessionId, connectWebSocket]);

  async function handleSubmitAction(
    prevState: {
      status: string;
      error?: string;
      transcript?: string;
      sessionId?: string;
    },
    formData: FormData
  ): Promise<{
    status: string;
    error?: string;
    transcript?: string;
    sessionId?: string;
  }> {
    setAgentStatus("starting");
    setCurrentGoal("Sending task to agent...");
    setLastAction("");
    setScreenshotData(null);
    setFinalResult(null);
    setAgentStep(0);
    setLastTranscript("");

    try {
      const response = await fetch("/api", {
        method: "POST",
        body: formData,
      });

      const transcript = decodeURIComponent(
        response.headers.get("X-Transcript") || ""
      );
      const receivedSessionId = response.headers.get("X-Session-Id");
      setLastTranscript(transcript);

      if (!response.ok) {
        const errorText = await response.text();
        toast.error(
          errorText || `Failed to start agent task (${response.status})`
        );
        setAgentStatus("error");
        setCurrentGoal(`Error starting task: ${response.statusText}`);
        return { status: "error", error: errorText, transcript };
      }

      const agentResponse = await response.json();
      const currentSessionId = agentResponse.session_id || receivedSessionId;

      if (!currentSessionId) {
        toast.error("Backend did not return a session ID.");
        setAgentStatus("error");
        setCurrentGoal("Error: Missing Session ID");
        return { status: "error", error: "Missing Session ID", transcript };
      }

      setSessionId(currentSessionId);
      setAgentStatus(agentResponse.status || "running");
      setCurrentGoal("Task submitted, waiting for agent...");

      return { status: "success", transcript, sessionId: currentSessionId };
    } catch (error: any) {
      console.error("Error submitting task form:", error);
      toast.error(`Submission failed: ${error.message || "Network error"}`);
      setAgentStatus("error");
      setCurrentGoal("Error communicating with backend");
      return {
        status: "error",
        error: error.message || "Unknown submission error",
      };
    }
  }

  function handleTextFormSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (
      !input.trim() ||
      isSubmitting ||
      (agentStatus !== "idle" &&
        agentStatus !== "error" &&
        agentStatus !== "done_success" &&
        agentStatus !== "done_failure")
    ) {
      console.log(
        "Submission prevented: Input empty, submitting, or agent busy."
      );
      if (agentStatus !== "idle") toast.info("Agent is currently busy.");
      return;
    }
    const formData = new FormData();
    formData.append("input", input);
    if (sessionId) formData.append("session_id", sessionId);
    formAction(formData);
  }

  useEffect(() => {
    function keyDown(e: KeyboardEvent) {
      if ((e.target as HTMLElement)?.closest("input, textarea")) {
        return;
      }
      if (e.key === "Enter") return inputRef.current?.focus();
      if (e.key === "Escape") return setInput("");
    }
    window.addEventListener("keydown", keyDown);
    return () => window.removeEventListener("keydown", keyDown);
  }, []);

  // --- NEW: Microphone Button Click Handler ---
  const handleMicClick = () => {
    if (vad.loading) {
      toast.error("Speech detection is still loading.");
      return;
    }
    if (
      agentStatus !== "idle" &&
      agentStatus !== "error" &&
      agentStatus !== "done_success" &&
      agentStatus !== "done_failure"
    ) {
      toast.info("Agent is currently busy.");
      return;
    }

    if (vad.listening) {
      console.log("Mic button clicked: pausing VAD");
      vad.pause(); // If already listening, stop it
    } else {
      console.log("Mic button clicked: starting VAD");
      // Clear previous transcript when starting new recording
      setLastTranscript("");
      vad.start(); // Start listening explicitly
    }
  };

  // --- Determine overall busy state ---
  const isAgentBusy =
    isSubmitting ||
    agentStatus === "running" ||
    agentStatus === "starting" ||
    agentStatus === "transcribing";

  return (
    <>
      <div className="w-full max-w-5xl mb-6 p-4 border border-neutral-300 dark:border-neutral-700 rounded-lg bg-white dark:bg-black min-h-[400px] flex flex-col md:flex-row gap-4 shadow-sm">
        <div className="flex-1 space-y-2 pr-4 border-r-0 md:border-r border-neutral-200 dark:border-neutral-800">
          <button onClick={() => setSessionId("test-123")}>
            Connect WS Test
          </button>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            Agent Status:
            <span
              className={`font-mono text-xs px-2 py-0.5 rounded font-medium ${
                agentStatus === "running"
                  ? "bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-200"
                  : agentStatus === "done_success"
                  ? "bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-200"
                  : agentStatus === "done_failure" || agentStatus === "paused"
                  ? "bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-200"
                  : agentStatus === "error" || agentStatus === "disconnected"
                  ? "bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-200"
                  : "bg-neutral-100 dark:bg-neutral-800 text-neutral-500 dark:text-neutral-400"
              }`}
            >
              {agentStatus}
            </span>
            {agentStatus === "running" && <LoadingIcon />}
          </h2>
          {sessionId && (
            <p className="text-xs text-neutral-500 dark:text-neutral-500">
              Session: {sessionId.substring(0, 8)}...
            </p>
          )}
          <div className="text-sm text-neutral-700 dark:text-neutral-300 space-y-1 pt-2">
            <p>
              <strong>Step:</strong> {agentStep > 0 ? agentStep : "N/A"}
            </p>
            <p>
              <strong>Goal:</strong>{" "}
              {currentGoal ||
                (agentStatus === "idle" ? "Waiting for task..." : "...")}
            </p>
            <p>
              <strong>Action:</strong> {lastAction || "None"}
            </p>
            {finalResult && (
              <p className="font-medium mt-3 pt-3 border-t border-neutral-200 dark:border-neutral-800">
                <strong>Result:</strong> {finalResult}
              </p>
            )}
          </div>
        </div>
        <div className="w-full md:w-1/2 lg:w-3/5 border border-neutral-200 dark:border-neutral-800 rounded-md overflow-hidden bg-neutral-100 dark:bg-neutral-900 flex items-center justify-center aspect-[16/10] md:aspect-auto">
          {screenshotData ? (
            <img
              src={`data:image/png;base64,${screenshotData}`}
              alt="Agent Screenshot"
              className="object-contain max-w-full max-h-full"
              width={1280}
              height={800}
            />
          ) : (
            <span className="text-neutral-500 text-sm p-4">
              {agentStatus === "running"
                ? "Waiting for screenshot..."
                : "No screenshot"}
            </span>
          )}
        </div>
      </div>

      <div className="pb-4 min-h-10" />

      <form
        className="rounded-full bg-neutral-100 dark:bg-neutral-900 flex items-center gap-1 w-full max-w-3xl border border-neutral-300 dark:border-neutral-700 hover:border-neutral-400 focus-within:border-neutral-500 dark:hover:border-neutral-600 dark:focus-within:border-neutral-500 shadow-sm transition-colors pr-2"
        onSubmit={handleTextFormSubmit}
      >
        <input
          type="text"
          className="bg-transparent focus:outline-none pl-4 pr-2 py-4 w-full placeholder:text-neutral-500 dark:placeholder:text-neutral-500 text-sm flex-grow"
          required
          placeholder={
            vad.listening
              ? "Listening via microphone..."
              : isAgentBusy
              ? "Agent is running..."
              : "Type a command or press the mic..."
          }
          value={input}
          onChange={(e) => setInput(e.target.value)}
          ref={inputRef}
          disabled={isAgentBusy || vad.userSpeaking || vad.listening}
        />
        {/* Submit button for text */}
        <button
          type="submit"
          className="p-2 text-neutral-600 hover:text-black dark:text-neutral-400 dark:hover:text-white disabled:opacity-40 rounded-full hover:bg-neutral-200 dark:hover:bg-neutral-800 transition-colors flex-shrink-0"
          disabled={isAgentBusy || !input.trim()}
          aria-label="Submit text command"
        >
          {isSubmitting || agentStatus === "starting" ? (
            <LoadingIcon />
          ) : (
            <EnterIcon />
          )}
        </button>
        {/* NEW Microphone Button */}
        <button
          type="button"
          onClick={handleMicClick}
          className={`p-2 rounded-full transition-colors flex-shrink-0 ${
            vad.listening
              ? "text-red-500 bg-red-100 dark:bg-red-900 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-800"
              : "text-blue-600 dark:text-blue-400 hover:bg-blue-100 dark:hover:bg-blue-900 disabled:opacity-40"
          }`}
          disabled={isAgentBusy || vad.loading}
          aria-label={vad.listening ? "Stop listening" : "Start listening"}
        >
          {vad.loading ? (
            <LoadingIcon />
          ) : vad.listening ? (
            <StopCircleIcon />
          ) : (
            <MicrophoneIcon />
          )}
        </button>
      </form>

      <div className="text-neutral-500 dark:text-neutral-500 pt-4 text-center max-w-xl text-balance min-h-28 text-xs space-y-2">
        {lastTranscript && (
          <p className="italic">You said: "{lastTranscript}"</p>
        )}

        {agentStatus === "idle" && !lastTranscript && (
          <>
            <p className="font-medium">Speak (press mic) or type a task.</p>
            {vad.loading ? (
              <p>Loading speech detection...</p>
            ) : (
              <p>
                Mic Status:{" "}
                {vad.listening
                  ? "Listening..."
                  : vad.userSpeaking
                  ? "Processing..."
                  : "Ready"}
              </p>
            )}
          </>
        )}
        {formSubmissionState.status === "error" && (
          <p className="text-red-500">
            Submission Error: {formSubmissionState.error}
          </p>
        )}
      </div>

      <div
        className={clsx(
          "fixed bottom-10 left-1/2 -translate-x-1/2 size-24 blur-2xl rounded-full bg-gradient-to-b from-blue-300 to-blue-500 dark:from-blue-700 dark:to-blue-900 -z-10 transition-all duration-300 ease-in-out",
          {
            "opacity-0 scale-50": vad.loading || agentStatus !== "idle",
            "opacity-30":
              !vad.loading &&
              !vad.userSpeaking &&
              !vad.listening &&
              agentStatus === "idle",
            "opacity-50 scale-100":
              vad.listening && !vad.userSpeaking && agentStatus === "idle",
            "opacity-70 scale-110": vad.userSpeaking && agentStatus === "idle",
          }
        )}
      />
    </>
  );
}

function A(props: React.AnchorHTMLAttributes<HTMLAnchorElement>) {
  return (
    <a
      {...props}
      className="text-neutral-600 dark:text-neutral-400 hover:underline font-medium"
    />
  );
}
