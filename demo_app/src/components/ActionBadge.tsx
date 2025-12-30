import React from "react";

type Props = {
  label: string;
  tone?: "rest" | "open" | "close" | "neutral";
};

const toneMap: Record<string, string> = {
  rest: "badge rest",
  open: "badge open",
  close: "badge close",
  neutral: "badge"
};

export function ActionBadge({ label, tone = "neutral" }: Props) {
  return (
    <div className={toneMap[tone] ?? toneMap.neutral}>
      <span>{label}</span>
    </div>
  );
}
