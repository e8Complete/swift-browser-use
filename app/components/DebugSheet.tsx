// app/components/DebugSheet.tsx
"use client";

import { useState, useRef, useEffect } from "react";
import clsx from "clsx";
import { Rnd } from "react-rnd";
import { MinimizeIcon, MaximizeIcon } from "@/lib/icons";

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

// Define dimensions for expanded/minimized states
const expandedWidth = 480;
const expandedHeightVh = 85;
const minimizedWidth = 256;
const minimizedHeight = 58;

interface DebugSheetProps {
  historyData: AgentHistoryData | null;
  defaultPosition?: { x: number; y: number };
}

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

export default function DebugSheet({
  historyData,
  defaultPosition,
}: DebugSheetProps) {
  const [activeTab, setActiveTab] = useState<DebugCategory>(tabs[0]);
  const [isMinimized, setIsMinimized] = useState(false);
  const [defaultHeight, setDefaultHeight] = useState(800);
  const [defaultPos, setDefaultPos] = useState({ x: 20, y: 20 });

  // Handle window dimensions after mount
  useEffect(() => {
    const calculateDimensions = () => {
      setDefaultHeight(
        Math.min(800, window.innerHeight * (expandedHeightVh / 100))
      );
      setDefaultPos({
        x: window.innerWidth - expandedWidth - 20,
        y: 20,
      });
    };

    calculateDimensions();
    window.addEventListener("resize", calculateDimensions);

    return () => window.removeEventListener("resize", calculateDimensions);
  }, []);

  const toggleMinimize = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsMinimized(!isMinimized);
  };

  const renderTabData = (tab: DebugCategory) => {
    if (!historyData) {
      return (
        <p className="text-sm text-gray-500 p-4 text-center">
          No data available.
        </p>
      );
    }

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
    <Rnd
      size={
        isMinimized
          ? { width: minimizedWidth, height: minimizedHeight }
          : { width: expandedWidth, height: defaultHeight }
      }
      minWidth={minimizedWidth}
      minHeight={minimizedHeight}
      maxWidth={"80vw"}
      maxHeight={"90vh"}
      default={
        defaultPosition
          ? {
              ...defaultPosition,
              width: expandedWidth,
              height: defaultHeight,
            }
          : {
              ...defaultPos,
              width: expandedWidth,
              height: defaultHeight,
            }
      }
      bounds="body"
      dragHandleClassName="drag-handle"
      enableResizing={!isMinimized}
      className="z-[9999]"
      style={{
        position: "fixed",
        visibility: "visible",
        pointerEvents: "auto",
      }}
    >
      <div
        className={clsx(
          "bg-white border border-gray-300 rounded-lg",
          "shadow-lg",
          "flex flex-col",
          "w-full h-full",
          "overflow-hidden"
        )}
      >
        <div className="drag-handle p-4 border-b border-gray-200 flex-shrink-0 rounded-t-lg cursor-move flex justify-between items-center bg-gray-50">
          <div>
            <h2 className="text-base font-semibold text-gray-800">
              Debugging Information
            </h2>
            {historyData && !isMinimized && (
              <p className="text-xs text-gray-500 mt-1">
                Snapshot: {historyData.snapshotTimestamp}
              </p>
            )}
          </div>
          <button
            onClick={toggleMinimize}
            className="p-1 rounded hover:bg-gray-200 text-gray-500 hover:text-gray-800"
            aria-label={isMinimized ? "Maximize" : "Minimize"}
            title={isMinimized ? "Maximize" : "Minimize"}
          >
            {isMinimized ? <MaximizeIcon /> : <MinimizeIcon />}
          </button>
        </div>

        {!isMinimized && (
          <>
            <div className="flex flex-wrap gap-1 border-b border-gray-200 px-3 py-2 bg-gray-50 flex-shrink-0">
              {tabs.map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  disabled={!historyData}
                  className={clsx(
                    "px-3 py-1.5 text-xs font-medium rounded-md focus:outline-none whitespace-nowrap transition-colors duration-150 ease-in-out",
                    !historyData
                      ? "text-gray-400 bg-gray-100 cursor-not-allowed"
                      : activeTab === tab
                      ? "bg-blue-600 text-white shadow-sm"
                      : "text-gray-600 hover:bg-gray-200"
                  )}
                >
                  {tab
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (l) => l.toUpperCase())}
                </button>
              ))}
            </div>

            <div className="flex-grow overflow-y-auto rounded-b-lg">
              {renderTabData(activeTab)}
            </div>
          </>
        )}
      </div>
    </Rnd>
  );
}
