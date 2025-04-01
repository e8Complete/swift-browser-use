import React from "react";

export function MinimizeIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M3 8H13"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function MaximizeIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <polyline
        points="4,10 4,12 12,12 12,4 4,4 4,6"
        stroke="currentColor"
        strokeWidth="2"
        fill="none"
      />
      <line x1="7" y1="8" x2="9" y2="8" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}
