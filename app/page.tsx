"use client";

import {
	IconCornerDownLeft,
	IconLoader2,
	IconMicrophone,
} from "@tabler/icons-react";
import clsx from "clsx";
import { assistant } from "@/app/actions";
import { useCallback, useEffect, useRef, useState, useTransition } from "react";
import { toast } from "sonner";

export default function Home() {
	const [isPending, startTransition] = useTransition();
	const [isRecording, setIsRecording] = useState(false);
	const recorder = useRef<MediaRecorder | null>(null);
	const chunks = useRef<Array<Blob>>([]);

	function dataAvailable(e: BlobEvent) {
		chunks.current.push(e.data);
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

				recorder.current.addEventListener("stop", () => {
					setIsRecording(false);
					startTransition(async () => {
						const blob = new Blob(chunks.current, {
							type: "audio/webm",
						});
						chunks.current = [];
						const base64 = await toBase64(blob);
						const { error, text } = await assistant(base64);
						if (error) {
							toast.error(error);
						} else {
							toast(text);
						}
					});
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
	}

	function handleMicButtonClick(e: React.MouseEvent) {
		e.preventDefault();

		if (isRecording) {
			stopRecording();
		} else {
			startRecording();
		}
	}

	return (
		<form className="rounded-full bg-neutral-200 flex items-center w-full max-w-3xl border border-transparent hover:border-neutral-300 focus-within:border-neutral-400 hover:focus-within:border-neutral-400">
			<button
				className={clsx(
					"rounded-full bg-white p-1 border border-neutral-300 drop-shadow m-3 hover:scale-110 active:scale-90 transition ease-in-out",
					{ "text-red-500": isRecording }
				)}
				onClick={handleMicButtonClick}
			>
				<IconMicrophone />
			</button>

			<input
				type="text"
				className="bg-transparent focus:outline-none py-3 w-full placeholder:text-neutral-700"
				required
				disabled={isRecording || isPending}
				placeholder="Ask me anything"
			/>

			<button
				type="submit"
				className={clsx("m-3 p-1 text-neutral-700 hover:text-black", {
					"animate-spin": isPending,
				})}
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
