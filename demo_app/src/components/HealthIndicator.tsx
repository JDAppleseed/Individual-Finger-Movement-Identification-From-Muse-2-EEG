import React from "react";

type Props = {
  score: number;
  lslConnected: boolean;
  mode: string;
};

export function HealthIndicator({ score, lslConnected, mode }: Props) {
  const pct = Math.round(score * 100);
  const status = lslConnected ? "Live" : "Offline";
  return (
    <div className="card">
      <div className="card-title">Signal Health</div>
      <div className="health-row">
        <div className="health-dot" />
        <div className="health-text">{pct}% · {status} · {mode.toUpperCase()}</div>
      </div>
    </div>
  );
}
