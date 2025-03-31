// app/elevenlabs/page.tsx
"use client";

import { useState, useRef, useCallback } from "react";
import { toast } from "sonner"; // Assuming you still want toast notifications for errors
import { LoadingIcon } from "@/lib/icons"; // Reuse the loading icon

export default function ElevenLabsTestPage() {
  const [text, setText] = useState<string>("Hello world! This is a test.");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const audioRef = useRef<HTMLAudioElement | null>(null); // Ref for the Audio object

  // Re-usable function to request speech synthesis and play it
  const requestSpeech = useCallback(async (textToSpeak: string) => {
    if (!textToSpeak.trim()) {
      toast.error("Please enter some text to speak.");
      return;
    }

    setIsLoading(true);
    console.log("Requesting speech for:", textToSpeak);

    // Stop and cleanup previous audio if it's playing
    if (audioRef.current) {
      if (!audioRef.current.paused) {
        audioRef.current.pause();
        console.log("Paused previous audio.");
      }
      // Revoke old object URL if src exists (important for cleanup)
      if (audioRef.current.src && audioRef.current.src.startsWith("blob:")) {
        URL.revokeObjectURL(audioRef.current.src);
        console.log("Revoked previous blob URL:", audioRef.current.src);
      }
      audioRef.current.removeAttribute("src"); // Remove src to be safe
    }

    try {
      const response = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToSpeak }),
      });

      if (!response.ok || !response.body) {
        const errorText = await response.text();
        throw new Error(
          `TTS request failed: ${response.status} ${response.statusText} - ${errorText}`
        );
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      console.log("Received audio blob, created URL:", audioUrl);

      // Ensure Audio object exists
      if (!audioRef.current) {
        audioRef.current = new Audio();
        console.log("Created new Audio element");
      } else {
        console.log("Reusing existing Audio element");
      }

      audioRef.current.src = audioUrl;

      // Add event listeners *before* playing
      audioRef.current.onended = () => {
        console.log("Playback finished for:", audioUrl);
        // It's safer to revoke here after playback finishes
        if (audioUrl) URL.revokeObjectURL(audioUrl);
      };
      audioRef.current.onerror = (e) => {
        console.error("Audio playback error:", e, "for URL:", audioUrl);
        toast.error("Failed to play agent speech.");
        if (audioUrl) URL.revokeObjectURL(audioUrl); // Cleanup on error too
      };

      console.log("Attempting to play audio...");
      await audioRef.current.play();
      console.log("Playback started.");
    } catch (error: any) {
      console.error("Error requesting or playing speech:", error);
      toast.error(`Speech synthesis failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, []); // requestSpeech depends on nothing external here, empty dependency array

  const handleSpeakClick = () => {
    requestSpeech(text);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-10 dark:text-white bg-white dark:bg-black">
      <h1 className="text-2xl font-bold mb-6">ElevenLabs TTS Test</h1>
      <div className="w-full max-w-md space-y-4">
        <label
          htmlFor="tts-input"
          className="block text-sm font-medium text-neutral-700 dark:text-neutral-300"
        >
          Text to Speak:
        </label>
        <textarea
          id="tts-input"
          rows={4}
          className="w-full p-2 border border-neutral-300 dark:border-neutral-700 rounded-md bg-white dark:bg-neutral-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text here..."
        />
        <button
          onClick={handleSpeakClick}
          disabled={isLoading || !text.trim()}
          className="w-full flex justify-center items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <>
              <LoadingIcon /> <span className="ml-2">Synthesizing...</span>
            </>
          ) : (
            "Speak Text"
          )}
        </button>
      </div>
      <p className="mt-6 text-xs text-neutral-500 dark:text-neutral-400">
        Ensure your <code>.env.local</code> has correct{" "}
        <code>ELEVENLABS_API_KEY</code> and <code>ELEVENLABS_VOICE_ID</code>.
        Check browser and server console for logs.
      </p>
    </div>
  );
}
