// app/debugging/page.tsx
"use client";

import { useState } from "react";
import DebugSheet, { AgentHistoryData } from "@/components/DebugSheet";

// Helper function to generate timestamps in the past
const pastTimestamp = (minutesAgo: number): string => {
  const date = new Date(Date.now() - minutesAgo * 60 * 1000);
  return date.toISOString();
};

// Example hardcoded history data for demonstration
const hardcodedHistory: AgentHistoryData = {
  snapshotTimestamp: new Date().toISOString(),
  urls: [
    { timestamp: pastTimestamp(5), value: "https://example.com/page1" },
    { timestamp: pastTimestamp(4), value: "https://example.com/page2" },
    { timestamp: pastTimestamp(3), value: "https://example.com/page3" },
  ],
  screenshots: [
    { timestamp: pastTimestamp(5), value: "screenshot_001.png" },
    { timestamp: pastTimestamp(3), value: "screenshot_002.png" },
  ],
  action_names: [
    { timestamp: pastTimestamp(5), value: "Click Search Button" },
    { timestamp: pastTimestamp(4), value: "Enter Search Query" },
    { timestamp: pastTimestamp(3), value: "Navigate to Results" },
  ],
  extracted_content: [
    {
      timestamp: pastTimestamp(4),
      value: ["Product Name: Example", "Price: $99.99", "Rating: 4.5 stars"],
    },
    {
      timestamp: pastTimestamp(3),
      value: ["Category: Electronics", "In Stock: Yes"],
    },
  ],
  errors: [
    {
      timestamp: pastTimestamp(4),
      value: "Failed to load image: Network timeout",
    },
  ],
  model_actions: [
    {
      timestamp: pastTimestamp(5),
      value: JSON.stringify({
        action: "click",
        selector: "#search-button",
        confidence: 0.95,
      }),
    },
    {
      timestamp: pastTimestamp(3),
      value: JSON.stringify({
        action: "type",
        selector: "#search-input",
        text: "example query",
        confidence: 0.98,
      }),
    },
  ],
};

export default function DebuggingPage() {
  // In a real scenario, this state would be populated via WebSocket or API call
  const [historyData] = useState<AgentHistoryData | null>(hardcodedHistory);

  return (
    <div className="flex h-screen bg-gray-100">
      {" "}
      {/* Main page background */}
      {/* Placeholder for main app content */}
      <div className="flex-grow p-6">
        <h1 className="text-2xl font-bold mb-4">Main Application Area</h1>
        <p className="text-gray-600">
          This space would contain the primary application UI. The debug sheet
          floats on the right.
        </p>
        <div className="h-[150vh] bg-gradient-to-b from-gray-100 to-gray-300 mt-10 rounded p-4">
          <p>Scrollable Content...</p>
        </div>
      </div>
      {/* Render the Debug Sheet Component */}
      <DebugSheet historyData={historyData} />
    </div>
  );
}
