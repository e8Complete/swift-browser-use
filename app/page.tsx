"use client";

import {
	IconCornerDownLeft,
	IconLoader2,
	IconMicrophone,
} from "@tabler/icons-react";
import clsx from "clsx";
import { assistant, type Messages } from "@/app/actions";
import { useCallback, useEffect, useRef, useState, useTransition } from "react";
import { toast } from "sonner";
import { useTTS } from "@cartesia/cartesia-js/react";

export default function Home() {
	const [isPending, startTransition] = useTransition();
	const [isRecording, setIsRecording] = useState(false);
	const [input, setInput] = useState("");
	const recorder = useRef<MediaRecorder | null>(null);
	const chunks = useRef<Array<Blob>>([]);
	const messages = useRef<Messages>([]);

	const tts = useTTS({
		apiKey: process.env.NEXT_PUBLIC_CARTESIA_API_KEY!,
		sampleRate: 44100,
	});

	function dataAvailable(e: BlobEvent) {
		chunks.current.push(e.data);
	}

	function submit(e?: React.FormEvent) {
		if (e) e.preventDefault();
		startTransition(async () => {
			let type;
			let data;

			if (isRecording) {
				type = "speech";
				stopRecording();
				const blob = new Blob(chunks.current, {
					type: "audio/webm",
				});
				chunks.current = [];
				data = await toBase64(blob);
			} else {
				type = "text";
				data = input;
			}

			const response = await assistant({
				type,
				data,
				prevMessages: messages.current,
			});

			if ("error" in response) {
				toast.error(response.error);
				return;
			}

			setInput(response.transcription);

			tts.buffer({
				model_id: "upbeat-moon",
				voice: {
					mode: "id",
					id: "00a77add-48d5-4ef6-8157-71e5437b282d",
				},
				transcript: response.text,
			});

			tts.play();

			toast(response.text, {
				duration: Math.max(response.text.length * 75, 5000),
			}); // TODO: better UI for showing messages or remove this

			messages.current.push({
				role: "user",
				content: response.transcription,
			});

			messages.current.push({
				role: "assistant",
				content: response.text,
			});
		});
	}

	const getRecorder = useCallback(() => {
		navigator.mediaDevices
			.getUserMedia({ audio: true })
			.then((stream) => {
				const mimeType = getSupportedMimeType();
				if (!mimeType) {
					return toast.error(
						"Your browser does not support audio recording."
					);
				}

				recorder.current = new MediaRecorder(stream, {
					mimeType,
				});

				recorder.current.addEventListener(
					"dataavailable",
					dataAvailable
				);
			})
			.catch(() => {
				return toast.error("Access to microphone was denied.");
			});
	}, []);

	useEffect(() => {
		getRecorder();
	}, [getRecorder]);

	function startRecording() {
		if (!recorder.current) getRecorder();
		if (!recorder.current) return;

		recorder.current.start();
		setIsRecording(true);
	}

	function stopRecording() {
		if (!recorder.current) return;
		recorder.current.stop();
		recorder.current.removeEventListener("dataavailable", dataAvailable);
		setIsRecording(false);
	}

	function handleMicButtonClick(e: React.MouseEvent) {
		e.preventDefault();

		if (isRecording) {
			submit();
		} else {
			startRecording();
		}
	}

	return (
		<form
			className="rounded-full bg-neutral-200 dark:bg-neutral-800 flex items-center w-full max-w-3xl border border-transparent hover:border-neutral-300 focus-within:border-neutral-400 hover:focus-within:border-neutral-400 dark:hover:border-neutral-700 dark:focus-within:border-neutral-600 dark:hover:focus-within:border-neutral-600"
			onSubmit={submit}
		>
			<button
				className={clsx("p-3 box-border group", {
					"text-red-500": isRecording,
				})}
				onClick={handleMicButtonClick}
				type="button"
			>
				<div className="rounded-full bg-white dark:bg-black border border-neutral-300 dark:border-neutral-700 drop-shadow group-hover:scale-110 group-active:scale-90 transition ease-in-out p-1">
					<IconMicrophone />
				</div>
			</button>

			<input
				type="text"
				className="bg-transparent focus:outline-none py-3 w-full placeholder:text-neutral-700 dark:placeholder:text-neutral-300"
				required
				disabled={isRecording || isPending}
				placeholder="Ask me anything"
				value={input}
				onChange={(e) => setInput(e.target.value)}
			/>

			<button
				type="submit"
				className={clsx(
					"p-4 text-neutral-700 hover:text-black dark:text-neutral-300 dark:hover:text-white",
					{
						"animate-spin": isPending,
					}
				)}
				disabled={isPending}
			>
				{isPending ? <IconLoader2 /> : <IconCornerDownLeft />}
			</button>
		</form>
	);
}

function toBase64(blob: Blob): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.readAsDataURL(blob);
		reader.onload = () => {
			if (typeof reader.result !== "string") return;
			resolve(reader.result);
		};
		reader.onerror = (error) => reject(error);
	});
}

const types = ["audio/webm", "video/mp4", "audio/mpeg", "audio/wav"];
function getSupportedMimeType() {
	return types.find((type) => MediaRecorder.isTypeSupported(type));
}
