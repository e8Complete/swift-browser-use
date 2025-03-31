import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";

const elevenLabsApiKey = process.env.ELEVENLABS_API_KEY;
const elevenLabsVoiceId = process.env.ELEVENLABS_VOICE_ID; // e.g., '21m00Tcm4TlvDq8ikWAM'

const schema = z.object({
  text: z.string().min(1, "Text cannot be empty"),
});

export async function POST(request: NextRequest) {
  if (!elevenLabsApiKey) {
    console.error("ELEVENLABS_API_KEY is not set.");
    return new NextResponse(
      "Server configuration error: Missing ElevenLabs API key.",
      { status: 500 }
    );
  }
  if (!elevenLabsVoiceId) {
    console.error("ELEVENLABS_VOICE_ID is not set.");
    return new NextResponse(
      "Server configuration error: Missing ElevenLabs Voice ID.",
      { status: 500 }
    );
  }

  let textToSpeak: string;
  try {
    const body = await request.json();
    const validatedData = schema.parse(body);
    textToSpeak = validatedData.text;
  } catch (error) {
    if (error instanceof z.ZodError) {
      return new NextResponse(
        `Invalid request body: ${error.errors
          .map((e) => e.message)
          .join(", ")}`,
        { status: 400 }
      );
    }
    console.error("Error parsing request body:", error);
    return new NextResponse("Invalid request body.", { status: 400 });
  }

  const elevenLabsUrl = `https://api.elevenlabs.io/v1/text-to-speech/${elevenLabsVoiceId}/stream`;
  const headers = {
    Accept: "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": elevenLabsApiKey,
  };
  const body = JSON.stringify({
    text: textToSpeak,
    model_id: "eleven_multilingual_v2", // Or your preferred model
    voice_settings: {
      stability: 0.5,
      similarity_boost: 0.75,
      // style: 0.0, // Optional: Adjust style exaggeration
      // use_speaker_boost: true // Optional: Boost speaker clarity
    },
  });

  try {
    console.log(
      `Requesting TTS from ElevenLabs for: "${textToSpeak.substring(0, 50)}..."`
    );
    const response = await fetch(elevenLabsUrl, {
      method: "POST",
      headers,
      body,
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error(`ElevenLabs API Error (${response.status}):`, errorData);
      return new NextResponse(`ElevenLabs API Error: ${response.statusText}`, {
        status: response.status,
      });
    }

    if (!response.body) {
      console.error("ElevenLabs response body is null.");
      return new NextResponse("Received empty response body from ElevenLabs.", {
        status: 500,
      });
    }

    // Stream the audio back to the client
    return new Response(response.body, {
      headers: {
        "Content-Type": "audio/mpeg", // Ensure this matches the 'Accept' header and ElevenLabs output
      },
    });
  } catch (error: any) {
    console.error("Error calling ElevenLabs API:", error);
    return new NextResponse(`Failed to call ElevenLabs API: ${error.message}`, {
      status: 503,
    }); // Service Unavailable
  }
}
