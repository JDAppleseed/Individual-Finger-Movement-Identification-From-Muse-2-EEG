import React, { useEffect, useState } from "react";
import { Activity, Clock, Wifi } from "lucide-react";

type Props = {
  action: string;
  finger: string;
  confidence: number;
  latencyMs: number;
  wsState: string;
  lastMessageAt: number | null;
};

function formatAgo(ms: number | null, now: number) {
  if (!ms) return "--";
  const delta = now - ms;
  if (delta < 1000) return "just now";
  return `${(delta / 1000).toFixed(1)}s ago`;
}

export default function HandOverlayHUD({ action, finger, confidence, latencyMs, wsState, lastMessageAt }: Props) {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div className="hand-hud">
      <div className="hud-title">Live Hand Telemetry</div>
      <div className="hud-row">
        <Activity size={14} />
        Action: <strong>{action}</strong>
      </div>
      <div className="hud-row">
        <Activity size={14} />
        Finger: <strong>{finger}</strong>
      </div>
      <div className="hud-row">
        <Activity size={14} />
        Confidence: <strong>{Math.round(confidence * 100)}%</strong>
      </div>
      <div className="hud-row">
        <Clock size={14} />
        Latency: <strong>{latencyMs.toFixed(1)} ms</strong>
      </div>
      <div className="hud-row">
        <Wifi size={14} />
        WS: <strong>{wsState}</strong>
      </div>
      <div className="hud-row">
        <Clock size={14} />
        Last tick: <strong>{formatAgo(lastMessageAt, now)}</strong>
      </div>
    </div>
  );
}
