"use client";

import clsx from "clsx";
import React, { useCallback, useRef, useState, useTransition } from "react";
import { toast } from "sonner";
import { EnterIcon, LoadingIcon } from "@/lib/icons";
import { usePlayer } from "@/lib/usePlayer";
import { useHotkeys } from "@/lib/useHotkeys";
import { track } from "@vercel/analytics";
import { useMicVAD, utils } from "@ricky0123/vad-react";

export default function Home() {
	const [isPending, startTransition] = useTransition();
	const [input, setInput] = useState("");
	const [latency, setLatency] = useState<number | null>(null);
	const [response, setResponse] = useState<string | null>(null);
	const messages = useRef<Array<object>>([]);
	const inputRef = useRef<HTMLInputElement>(null);
	const player = usePlayer();

	const vad = useMicVAD({
		startOnLoad: true,
		onSpeechEnd: (audio) => {
			player.stop();
			const wav = utils.encodeWAV(audio);
			const blob = new Blob([wav], { type: "audio/wav" });
			submit(blob);
		},
		workletURL: "/vad.worklet.bundle.min.js",
		modelURL: "/silero_vad.onnx",
		positiveSpeechThreshold: 0.7,
		minSpeechFrames: 5,
		ortConfig(ort) {
			ort.env.wasm.wasmPaths = {
				"ort-wasm-simd-threaded.wasm": "/ort-wasm-simd-threaded.wasm",
				"ort-wasm-simd.wasm": "/ort-wasm-simd.wasm",
				"ort-wasm.wasm": "/ort-wasm.wasm",
				"ort-wasm-threaded.wasm": "/ort-wasm-threaded.wasm",
			};
		},
	});

	useHotkeys({
		enter: () => inputRef.current?.focus(),
		escape: () => setInput(""),
	});

	const submit = useCallback(
		(data: string | Blob) => {
			startTransition(async () => {
				const formData = new FormData();

				if (typeof data === "string") {
					formData.append("input", data);
					track("Text input");
				} else {
					formData.append("input", data, "audio.wav");
					track("Speech input");
				}

				for (const message of messages.current) {
					formData.append("message", JSON.stringify(message));
				}

				const submittedAt = Date.now();

				const response = await fetch("/api", {
					method: "POST",
					body: formData,
				});

				const transcript = decodeURIComponent(
					response.headers.get("X-Transcript") || ""
				);
				const text = decodeURIComponent(
					response.headers.get("X-Response") || ""
				);

				if (!response.ok || !transcript || !text || !response.body) {
					const error =
						(await response.text()) || "An error occurred.";
					toast.error(error);
					return;
				}

				setLatency(Date.now() - submittedAt);
				player.play(response.body);
				setInput(transcript);
				setResponse(text);

				messages.current.push(
					{
						role: "user",
						content: transcript,
					},
					{
						role: "assistant",
						content: text,
					}
				);
			});
		},
		[player]
	);

	function handleFormSubmit(e: React.FormEvent) {
		e.preventDefault();
		submit(input);
	}

	return (
		<>
			<div className="pb-4 min-h-28" />

			<form
				className="rounded-full bg-neutral-200/80 dark:bg-neutral-800/80 flex items-center w-full max-w-3xl border border-transparent hover:border-neutral-300 focus-within:border-neutral-400 hover:focus-within:border-neutral-400 dark:hover:border-neutral-700 dark:focus-within:border-neutral-600 dark:hover:focus-within:border-neutral-600"
				onSubmit={handleFormSubmit}
			>
				<input
					type="text"
					className="bg-transparent focus:outline-none p-4 w-full placeholder:text-neutral-600 dark:placeholder:text-neutral-400"
					required
					placeholder="Ask me anything"
					value={input}
					onChange={(e) => setInput(e.target.value)}
					ref={inputRef}
				/>

				<button
					type="submit"
					className="p-4 text-neutral-700 hover:text-black dark:text-neutral-300 dark:hover:text-white"
					disabled={isPending}
					aria-label="Submit"
				>
					{isPending ? <LoadingIcon /> : <EnterIcon />}
				</button>
			</form>

			<div className="text-neutral-400 dark:text-neutral-600 pt-4 text-center max-w-xl text-balance min-h-28 space-y-4">
				{response && (
					<p>
						{response}
						<span className="text-xs font-mono text-neutral-300 dark:text-neutral-700">
							{" "}
							({latency}ms)
						</span>
					</p>
				)}

				{!response && (
					<>
						<p>
							A fast, open-source voice assistant powered by{" "}
							<A href="https://groq.com">Groq</A>,{" "}
							<A href="https://cartesia.ai">Cartesia</A>,{" "}
							<A href="https://www.vad.ricky0123.com/">VAD</A>,
							and <A href="https://vercel.com">Vercel</A>.{" "}
							<A
								href="https://github.com/ai-ng/swift"
								target="_blank"
							>
								Learn more
							</A>
							.
						</p>

						{vad.loading ? (
							<p>Loading speech detection...</p>
						) : vad.errored ? (
							<p>Failed to load speech detection.</p>
						) : (
							<p>Start talking to chat.</p>
						)}
					</>
				)}
			</div>

			<div
				className={clsx(
					"absolute size-36 blur-3xl rounded-full bg-gradient-to-b from-red-200 to-red-400 dark:from-red-600 dark:to-red-800 -z-50 transition ease-in-out",
					{
						"opacity-0": vad.loading || vad.errored,
						"opacity-30":
							!vad.loading && !vad.errored && !vad.userSpeaking,
						"opacity-100 scale-110": vad.userSpeaking,
					}
				)}
			/>
		</>
	);
}

function A(props: any) {
	return (
		<a
			{...props}
			className="text-neutral-500 dark:text-neutral-500 hover:underline font-medium"
		/>
	);
}
