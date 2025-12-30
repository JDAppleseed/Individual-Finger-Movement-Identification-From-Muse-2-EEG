import React from "react";

const colorMap: Record<string, string> = {
  REST: "#6b7280",
  OPEN: "#22c55e",
  CLOSE: "#ef4444"
};

type Props = {
  actions: string[];
};

export function TimelineStrip({ actions }: Props) {
  return (
    <div className="timeline">
      {actions.map((action, idx) => (
        <span
          key={`${action}-${idx}`}
          className="timeline-bar"
          style={{ backgroundColor: colorMap[action] || "#94a3b8" }}
        />
      ))}
    </div>
  );
}
