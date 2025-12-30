import React from "react";

type Props = {
  confidence: number;
  threshold: number;
};

export function ThresholdGauge({ confidence, threshold }: Props) {
  const conf = Math.round(confidence * 100);
  const th = Math.round(threshold * 100);
  return (
    <div className="card">
      <div className="card-title">Threshold</div>
      <div className="gauge">
        <div className="gauge-track" />
        <div className="gauge-mark" style={{ left: `${th}%` }} />
        <div className="gauge-pointer" style={{ left: `${conf}%` }} />
      </div>
      <div className="card-sub">conf {conf}% / thresh {th}%</div>
    </div>
  );
}
