import React from "react";

type Props = {
  value: number;
  label: string;
};

export function ConfidenceMeter({ value, label }: Props) {
  const pct = Math.round(value * 100);
  return (
    <div className="card">
      <div className="card-title">{label}</div>
      <div className="meter">
        <div className="meter-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="card-sub">{pct}%</div>
    </div>
  );
}
