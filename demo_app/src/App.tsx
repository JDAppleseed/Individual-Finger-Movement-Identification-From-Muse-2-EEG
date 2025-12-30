import React, { useEffect, useMemo, useState } from "react";
import "./styles.css";
import { connectWS } from "./api/ws";
import type { InferenceTick, StatusMessage } from "./types/schema";
import { ActionBadge } from "./components/ActionBadge";
import { ConfidenceMeter } from "./components/ConfidenceMeter";
import { UncertaintyDial } from "./components/UncertaintyDial";
import { ThresholdGauge } from "./components/ThresholdGauge";
import { LatencyChip } from "./components/LatencyChip";
import { HealthIndicator } from "./components/HealthIndicator";
import { TimelineStrip } from "./components/TimelineStrip";
import { ControlPanel } from "./components/ControlPanel";
import Waves from "./components/Waves";

const WS_URL = "ws://127.0.0.1:8008/stream";
const API_URL = "http://127.0.0.1:8008/control";

export default function App() {
  const [tick, setTick] = useState<InferenceTick | null>(null);
  const [status, setStatus] = useState<StatusMessage | null>(null);
  const [timeline, setTimeline] = useState<string[]>([]);
  const [mode, setMode] = useState<string>("replay");
  const [fps, setFps] = useState<number>(20);
  const [mcPasses, setMcPasses] = useState<number>(10);
  const [replayPath, setReplayPath] = useState<string>("eeg_windows.npz");
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    const socket = connectWS(WS_URL, {
      onTick: (msg) => {
        setTick(msg);
        setTimeline((prev) => {
          const next = [...prev, msg.prediction.action_name];
          return next.slice(-120);
        });
      },
      onStatus: (msg) => setStatus(msg)
    });
    setWs(socket);
    return () => socket.close();
  }, []);

  const tone = useMemo(() => {
    const action = tick?.prediction.action_name ?? "REST";
    if (action === "OPEN") return "open";
    if (action === "CLOSE") return "close";
    return "rest";
  }, [tick]);

  const calibrationLoaded = tick?.diagnostics.notes.includes("calibration_loaded") ?? false;

  async function handleStart() {
    await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode,
        replay_path: replayPath,
        fps,
        device: "cpu",
        mc_passes: mcPasses
      })
    });
  }

  async function handleStop() {
    await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "idle" })
    });
  }

  return (
    <div className="app">
    <div className="bg-waves">
      <div className="bg-waves-inner">
        <Waves
          lineColor="#2a03c6ff"
          backgroundColor="#0c0119ff"
          waveSpeedX={0.085}
          waveSpeedY={0.055}
          waveAmpX={25}
          waveAmpY={25}
          friction={0.57}
          tension={0.035}
          maxCursorMove={120}
          xGap={12}
          yGap={36}
        />
      </div>
    </div>
      <header className="hero">
        <div>
          <h1>EEG Finger Control Demo</h1>
          <p>Live inference, uncertainty, and safety gating for CNN/LSTM.</p>
        </div>
        <div className="status-row">
          <span className="status-row-debug-note">Status bar</span>
          <LatencyChip value={tick?.diagnostics.latency_ms ?? 0} />
          <div className="chip">Calibration {calibrationLoaded ? "Loaded" : "None"}</div>
          <div className="chip">WS {ws?.readyState === 1 ? "Connected" : "Disconnected"}</div>
        </div>
      </header>

      <main className="grid">
        <section className="panel">
          <ActionBadge label={tick?.prediction.action_name ?? "IDLE"} tone={tone} />
          <div className="finger-badge">
            Finger: {tick?.prediction.finger_name ?? "NONE"}
          </div>
          <ConfidenceMeter label="Action Confidence" value={tick?.prediction.action_confidence ?? 0} />
          <ConfidenceMeter label="Finger Confidence" value={tick?.prediction.finger_confidence ?? 0} />
        </section>

        <section className="panel">
          <UncertaintyDial value={tick?.prediction.action_uncertainty ?? 0} />
          <ThresholdGauge
            confidence={tick?.prediction.action_confidence ?? 0}
            threshold={tick?.safety.adaptive_threshold ?? 0.75}
          />
          <HealthIndicator
            score={tick?.diagnostics.health_score ?? 0}
            lslConnected={tick?.diagnostics.lsl_connected ?? false}
            mode={tick?.mode ?? "idle"}
          />
          <div className="chip">Artifact suppression: {String(tick?.diagnostics.artifact_suppression ?? "n/a")}</div>
        </section>

        <section className="panel">
          <ControlPanel
            mode={mode}
            onModeChange={setMode}
            onStart={handleStart}
            onStop={handleStop}
            replayPath={replayPath}
            onReplayPathChange={setReplayPath}
            fps={fps}
            onFpsChange={setFps}
            mcPasses={mcPasses}
            onMcPassesChange={setMcPasses}
          />
        </section>

        <section className="panel full">
          <div className="card-title">Timeline (last 30s)</div>
          <TimelineStrip actions={timeline} />
          <div className="status-text">
            {status?.message ?? "Ready"}
          </div>
        </section>
      </main>
    </div>
  );
}
