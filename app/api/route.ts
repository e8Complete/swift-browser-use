import Groq from "groq-sdk";
import OpenAI from "openai";
import { headers } from "next/headers";
import { z } from "zod";
import { zfd } from "zod-form-data";

const groqApiKey = process.env.GROQ_API_KEY;
const openaiApiKey = process.env.OPENAI_API_KEY;
const sttProvider = process.env.STT_PROVIDER || 'groq';

let groq: Groq | null = null;
if (sttProvider === 'groq' && groqApiKey) {
    groq = new Groq({ apiKey: groqApiKey });
}

let openai: OpenAI | null = null;
if (sttProvider === 'openai' && openaiApiKey) {
    openai = new OpenAI({ apiKey: openaiApiKey });
}

if (sttProvider === 'groq' && !groq) {
    console.error("STT_PROVIDER is 'groq' but GROQ_API_KEY is missing.");
}
if (sttProvider === 'openai' && !openai) {
    console.error("STT_PROVIDER is 'openai' but OPENAI_API_KEY is missing.");
}

const schema = zfd.formData({
    input: z.union([zfd.text(), zfd.file()]),
    message: zfd.repeatableOfType(
        zfd.json(z.object({
			role: z.enum(["user", "assistant"]),
			content: z.string()
		}))
    ).optional(),
    session_id: zfd.text().optional(),
});

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL;
if (!PYTHON_BACKEND_URL) {
    console.warn("WARN: PYTHON_BACKEND_URL environment variable not set. Defaulting to http://localhost:8000");
}
const agentApiEndpoint = `${PYTHON_BACKEND_URL || "http://localhost:8000"}/agent/task`;


export async function POST(request: Request) {
	console.time("transcribe " + request.headers.get("x-vercel-id") || "local");

	console.time("api_route_total");
    const { data, success } = schema.safeParse(await request.formData());
    if (!success)
		return new Response("Invalid form data", { status: 400 });

    // 1. Get Transcript
	console.time("transcribe");
    const transcript = await getTranscript(data.input);
    if (!transcript) return new Response(`Could not transcribe audio using ${sttProvider}`, { status: 400 });
    console.timeEnd("transcribe");
    console.log(`Transcript (${sttProvider}): ${transcript}`);

	// 2. Call Python Backend to start/add task
    console.time("agent_backend_call");
    let agentResponseData;
    try {
        console.log(`Calling agent backend: ${agentApiEndpoint}`);
        const agentTaskResponse = await fetch(agentApiEndpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            // Send the transcript as the 'task' and include session_id if present
            body: JSON.stringify({
                task: transcript,
                session_id: data.session_id // Pass session ID if it exists
            }),
        });

        agentResponseData = await agentTaskResponse.json();

        if (!agentTaskResponse.ok) {
             console.error(`Agent backend error (${agentTaskResponse.status}):`, agentResponseData);
             return new Response(agentResponseData.detail || `Agent backend error: ${agentTaskResponse.statusText}`, { status: agentTaskResponse.status });
        }
         console.info("Agent task initiated:", agentResponseData);

    } catch (error: any) {
        console.error("Failed to call agent backend:", error);
        return new Response(`Failed to communicate with agent backend: ${error.message}`, { status: 503 }); // Service Unavailable
    } finally {
        console.timeEnd("agent_backend_call");
    }


    // 4. Return JSON response from the agent backend call
    // The actual agent updates will come via WebSocket on the client-side.
    console.timeEnd("api_route_total");
	return new Response(JSON.stringify(agentResponseData), {
		status: 200, // OK - signifies the request to start was accepted
		headers: {
			"Content-Type": "application/json",
			"X-Transcript": encodeURIComponent(transcript), // Send transcript back for display
            "X-Session-Id": agentResponseData.session_id, // Ensure session ID is in headers too
		},
	});
}

function location() {
	const headersList = headers();

	const country = headersList.get("x-vercel-ip-country");
	const region = headersList.get("x-vercel-ip-country-region");
	const city = headersList.get("x-vercel-ip-city");

	if (!country || !region || !city) return "unknown";

	return `${city}, ${region}, ${country}`;
}

function time() {
	return new Date().toLocaleString("en-US", {
		timeZone: headers().get("x-vercel-ip-timezone") || undefined,
	});
}

async function getTranscript(input: string | File): Promise<string | null> {
    if (typeof input === "string") {
        return input.trim() || null;
    }

    if (sttProvider === 'openai') {
        if (!openai) return Promise.reject(new Error("OpenAI client not initialized. Check API key."));
        console.log("Using OpenAI STT");
        try {
            const transcription = await openai.audio.transcriptions.create({
                file: input,
                model: "whisper-1", // OpenAI's primary whisper model
                response_format: "text",
            });
             // OpenAI SDK returns the text directly when response_format is 'text'
            return (transcription as unknown as string).trim() || null;
        } catch (error) {
            console.error("OpenAI transcription error:", error);
            return null;
        }
    } else if (sttProvider === 'groq') {
         if (!groq) return Promise.reject(new Error("Groq client not initialized. Check API key."));
        console.log("Using Groq STT");
        try {
            const { text } = await groq.audio.transcriptions.create({
                file: input,
                model: "whisper-large-v3", // Groq uses specific model names
            });
            return text.trim() || null;
        } catch (error) {
            console.error("Groq transcription error:", error);
            return null;
        }
    } else {
        console.error(`Unsupported STT_PROVIDER: ${sttProvider}`);
        return Promise.reject(new Error(`Unsupported STT_PROVIDER: ${sttProvider}`));
    }
}
