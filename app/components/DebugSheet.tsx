// app/components/DebugSheet.tsx
"use client";

import { useState } from "react";
import clsx from "clsx";

// Define structure for individual data points with timestamps
export interface TimestampedData<T = string> {
  timestamp: string;
  value: T;
}

// Define the structure for our history data with timestamped arrays
export interface AgentHistoryData {
  snapshotTimestamp: string; // Timestamp of the overall history snapshot
  urls: TimestampedData<string>[];
  screenshots: TimestampedData<string>[]; // List of paths/identifiers
  action_names: TimestampedData<string>[];
  extracted_content: TimestampedData<string[]>[]; // Content might be multi-line
  errors: TimestampedData<string>[];
  model_actions: TimestampedData<string>[]; // Store as stringified JSON or descriptive strings
}

// Define the categories for tabs
type DebugCategory = keyof Omit<AgentHistoryData, "snapshotTimestamp">;

const tabs: DebugCategory[] = [
  "urls",
  "screenshots",
  "action_names",
  "extracted_content",
  "errors",
  "model_actions",
];

// Component to render individual data items with timestamps
const DataItem = ({
  timestamp,
  children,
}: {
  timestamp: string;
  children: React.ReactNode;
}) => (
  <div className="border-b border-gray-200 py-2 px-3 last:border-b-0">
    <p className="text-xs text-gray-500 mb-1">{timestamp}</p>
    <div>{children}</div>
  </div>
);

interface DebugSheetProps {
  historyData: AgentHistoryData | null;
}

export default function DebugSheet({ historyData }: DebugSheetProps) {
  const [activeTab, setActiveTab] = useState<DebugCategory>(tabs[0]);

  if (!historyData) {
    return null; // Don't render anything if no data is available
  }

  const renderTabData = (tab: DebugCategory) => {
    const data = historyData[tab];

    if (!data || data.length === 0) {
      return (
        <p className="text-sm text-gray-500 p-4 text-center">
          No data recorded for this category.
        </p>
      );
    }

    return (
      <div className="space-y-0">
        {data.map((item, index) => (
          <DataItem key={`${tab}-${index}`} timestamp={item.timestamp}>
            {(() => {
              switch (tab) {
                case "urls":
                  return (
                    <a
                      href={item.value as string}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-blue-600 hover:underline break-all"
                    >
                      {item.value as string}
                    </a>
                  );
                case "screenshots":
                  return (
                    <span className="text-sm font-mono">
                      {item.value as string}
                    </span>
                  );
                case "action_names":
                  return (
                    <span className="text-sm font-semibold">
                      {item.value as string}
                    </span>
                  );
                case "extracted_content":
                  return (
                    <pre className="text-xs bg-gray-50 p-1.5 rounded overflow-auto whitespace-pre-wrap">
                      {(item.value as string[]).join("\n")}
                    </pre>
                  );
                case "errors":
                  return (
                    <span className="text-sm text-red-600">
                      {item.value as string}
                    </span>
                  );
                case "model_actions":
                  try {
                    const parsed = JSON.parse(item.value as string);
                    return (
                      <pre className="text-xs bg-gray-50 p-1.5 rounded overflow-auto">
                        {JSON.stringify(parsed, null, 2)}
                      </pre>
                    );
                  } catch {
                    return (
                      <pre className="text-xs bg-gray-50 p-1.5 rounded overflow-auto">
                        {item.value as string}
                      </pre>
                    );
                  }
                default:
                  return <span className="text-sm">{String(item.value)}</span>;
              }
            })()}
          </DataItem>
        ))}
      </div>
    );
  };

  return (
    <div
      className={clsx(
        "fixed right-4 top-4 bottom-4",
        "w-[480px]",
        "bg-white border border-gray-200 rounded-lg",
        "shadow-2xl",
        "flex flex-col",
        "h-[calc(100vh-2rem)]",
        "z-20"
      )}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-200 flex-shrink-0 rounded-t-lg">
        <h2 className="text-lg font-semibold text-gray-800">
          Debugging Information
        </h2>
        <p className="text-xs text-gray-500 mt-1">
          Snapshot Time: {historyData.snapshotTimestamp}
        </p>
      </div>

      {/* Horizontal Tabs (with wrapping) */}
      <div className="flex flex-wrap gap-1 border-b border-gray-200 px-3 py-2 bg-gray-50 flex-shrink-0">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={clsx(
              "px-3 py-1.5 text-xs font-medium rounded-md focus:outline-none whitespace-nowrap transition-colors duration-150 ease-in-out",
              activeTab === tab
                ? "bg-blue-600 text-white shadow-sm"
                : "text-gray-600 hover:bg-gray-200"
            )}
          >
            {tab.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
          </button>
        ))}
      </div>

      {/* Tab Content Area */}
      <div className="flex-grow overflow-y-auto rounded-b-lg">
        {renderTabData(activeTab)}
      </div>
    </div>
  );
}
