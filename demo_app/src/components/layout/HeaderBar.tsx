import React, { useMemo, useState } from "react";
import { Activity, Bolt, Cable, Timer, Wifi } from "lucide-react";
import { shallow } from "zustand/shallow";
import { buildControlPayload, useControlMutation } from "../../api/client";
import { useDemoStore } from "../../state/useDemoStore";

function formatAgo(ms: number | null, now: number) {
  if (!ms) return "--";
  const delta = now - ms;
  if (delta < 1000) return "just now";
  return `${(delta / 1000).toFixed(1)}s ago`;
}

export default function HeaderBar() {
  const [now, setNow] = useState(Date.now());

  React.useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const {
    backendReachable,
    wsState,
    lastWsMessageAt,
    tick,
    mode,
    replayPath,
    fps,
    mcPasses,
    smoothingEnabled,
    smoothingMethod,
    smoothingWindow,
    hysteresisEnabled,
    hysteresisFrames,
    thresholdAction,
    thresholdFinger,
    adjacencyEnabled,
    setMode,
    setReplayPath,
    setFps,
    setSmoothingEnabled,
    setThresholdAction,
    setThresholdFinger
  } = useDemoStore(
    (state) => ({
      backendReachable: state.backendReachable,
      wsState: state.wsState,
      lastWsMessageAt: state.lastWsMessageAt,
      tick: state.tick,
      mode: state.mode,
      replayPath: state.replayPath,
      fps: state.fps,
      mcPasses: state.mcPasses,
      smoothingEnabled: state.smoothingEnabled,
      smoothingMethod: state.smoothingMethod,
      smoothingWindow: state.smoothingWindow,
      hysteresisEnabled: state.hysteresisEnabled,
      hysteresisFrames: state.hysteresisFrames,
      thresholdAction: state.thresholdAction,
      thresholdFinger: state.thresholdFinger,
      adjacencyEnabled: state.adjacencyEnabled,
      setMode: state.setMode,
      setReplayPath: state.setReplayPath,
      setFps: state.setFps,
      setSmoothingEnabled: state.setSmoothingEnabled,
      setThresholdAction: state.setThresholdAction,
      setThresholdFinger: state.setThresholdFinger
    }),
    shallow
  );

  const controlMutation = useControlMutation();

  const controlState = useMemo(
    () => ({
      mode,
      replayPath,
      fps,
      mcPasses,
      smoothingEnabled,
      smoothingMethod,
      smoothingWindow,
      hysteresisEnabled,
      hysteresisFrames,
      thresholdAction,
      thresholdFinger,
      adjacencyEnabled
    }),
    [
      mode,
      replayPath,
      fps,
      mcPasses,
      smoothingEnabled,
      smoothingMethod,
      smoothingWindow,
      hysteresisEnabled,
      hysteresisFrames,
      thresholdAction,
      thresholdFinger,
      adjacencyEnabled
    ]
  );

  const action = tick?.prediction.action_name ?? "REST";
  const finger = tick?.prediction.finger_name ?? "NONE";
  const latency = tick?.diagnostics.latency_ms ?? 0;

  const wsLabel = wsState === "open" ? "Live" : wsState.toUpperCase();
  const healthLabel = backendReachable === null ? "Checking" : backendReachable ? "Online" : "Offline";

  const handleDemoMode = () => {
    setMode("replay");
    setReplayPath("eeg_windows.npz");
    setFps(20);
    setSmoothingEnabled(true);
    setThresholdAction(0.75);
    setThresholdFinger(0.75);

    controlMutation.mutate(
      buildControlPayload(controlState, {
        mode: "replay",
        replay_path: "eeg_windows.npz",
        fps: 20,
        smoothing_enabled: true,
        threshold_action: 0.75,
        threshold_finger: 0.75
      })
    );
  };

  return (
    <header className="header-bar">
      <div className="header-left">
        <div className="title-stack">
          <p className="eyebrow">EEG Neural Control</p>
          <h1>Realtime Finger Intent Dashboard</h1>
          <p className="subtitle">Live inference, safety gating, and network introspection.</p>
        </div>
        <div className="status-pills">
          <div className={`status-pill ${backendReachable ? "ok" : "warn"}`}>
            <Cable size={16} />
            Backend: {healthLabel}
          </div>
          <div className={`status-pill ${wsState === "open" ? "ok" : "warn"}`}>
            <Wifi size={16} />
            WS: {wsLabel}
          </div>
          <div className="status-pill">
            <Timer size={16} />
            Last tick: {formatAgo(lastWsMessageAt, now)}
          </div>
        </div>
      </div>

      <div className="header-right">
        <div className="header-metrics">
          <div className="metric-chip">
            <Activity size={16} />
            Action: {action}
          </div>
          <div className="metric-chip">
            <Bolt size={16} />
            Finger: {finger}
          </div>
          <div className="metric-chip">
            <Timer size={16} />
            {latency.toFixed(1)} ms
          </div>
        </div>
        <button className="btn primary" onClick={handleDemoMode}>
          Demo Mode
        </button>
      </div>
    </header>
  );
}
