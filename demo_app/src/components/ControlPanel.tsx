import React from "react";

type Props = {
  mode: string;
  onModeChange: (mode: string) => void;
  onStart: () => void;
  onStop: () => void;
  replayPath: string;
  onReplayPathChange: (value: string) => void;
  fps: number;
  onFpsChange: (value: number) => void;
  mcPasses: number;
  onMcPassesChange: (value: number) => void;
  smoothingEnabled: boolean;
  onSmoothingEnabledChange: (value: boolean) => void;
  smoothingMethod: string;
  onSmoothingMethodChange: (value: string) => void;
  smoothingWindow: number;
  onSmoothingWindowChange: (value: number) => void;
  hysteresisEnabled: boolean;
  onHysteresisEnabledChange: (value: boolean) => void;
  hysteresisFrames: number;
  onHysteresisFramesChange: (value: number) => void;
  thresholdAction: number;
  onThresholdActionChange: (value: number) => void;
  thresholdFinger: number;
  onThresholdFingerChange: (value: number) => void;
  adjacencyEnabled: boolean;
  onAdjacencyEnabledChange: (value: boolean) => void;
};

export function ControlPanel({
  mode,
  onModeChange,
  onStart,
  onStop,
  replayPath,
  onReplayPathChange,
  fps,
  onFpsChange,
  mcPasses,
  onMcPassesChange,
  smoothingEnabled,
  onSmoothingEnabledChange,
  smoothingMethod,
  onSmoothingMethodChange,
  smoothingWindow,
  onSmoothingWindowChange,
  hysteresisEnabled,
  onHysteresisEnabledChange,
  hysteresisFrames,
  onHysteresisFramesChange,
  thresholdAction,
  onThresholdActionChange,
  thresholdFinger,
  onThresholdFingerChange,
  adjacencyEnabled,
  onAdjacencyEnabledChange
}: Props) {
  return (
    <div className="card control">
      <div className="card-title">Controls</div>
      <div className="control-row">
        <button className={mode === "replay" ? "btn active" : "btn"} onClick={() => onModeChange("replay")}>Replay</button>
        <button className={mode === "live" ? "btn active" : "btn"} onClick={() => onModeChange("live")}>Live</button>
      </div>
      <div className="control-row">
        <button className="btn primary" onClick={onStart}>Start</button>
        <button className="btn" onClick={onStop}>Stop</button>
      </div>
      <label className="control-label">Replay file</label>
      <input className="input" value={replayPath} onChange={(e) => onReplayPathChange(e.target.value)} />
      <label className="control-label">FPS: {fps}</label>
      <input className="slider" type="range" min={5} max={30} value={fps} onChange={(e) => onFpsChange(Number(e.target.value))} />
      <label className="control-label">MC passes: {mcPasses}</label>
      <input className="slider" type="range" min={5} max={30} value={mcPasses} onChange={(e) => onMcPassesChange(Number(e.target.value))} />

      <div className="control-row">
        <label className="control-label">
          <input type="checkbox" checked={smoothingEnabled} onChange={(e) => onSmoothingEnabledChange(e.target.checked)} />
          Smoothing
        </label>
        <label className="control-label">
          <input type="checkbox" checked={hysteresisEnabled} onChange={(e) => onHysteresisEnabledChange(e.target.checked)} />
          Hysteresis
        </label>
        <label className="control-label">
          <input type="checkbox" checked={adjacencyEnabled} onChange={(e) => onAdjacencyEnabledChange(e.target.checked)} />
          Adjacency assist
        </label>
      </div>
      <label className="control-label">Method</label>
      <select className="input" value={smoothingMethod} onChange={(e) => onSmoothingMethodChange(e.target.value)}>
        <option value="vote">Vote</option>
        <option value="ema">EMA</option>
      </select>
      <label className="control-label">Window N: {smoothingWindow}</label>
      <input className="slider" type="range" min={3} max={15} value={smoothingWindow} onChange={(e) => onSmoothingWindowChange(Number(e.target.value))} />
      <label className="control-label">Hysteresis frames: {hysteresisFrames}</label>
      <input className="slider" type="range" min={2} max={8} value={hysteresisFrames} onChange={(e) => onHysteresisFramesChange(Number(e.target.value))} />
      <label className="control-label">Action threshold: {Math.round(thresholdAction * 100)}%</label>
      <input className="slider" type="range" min={50} max={95} value={Math.round(thresholdAction * 100)} onChange={(e) => onThresholdActionChange(Number(e.target.value) / 100)} />
      <label className="control-label">Finger threshold: {Math.round(thresholdFinger * 100)}%</label>
      <input className="slider" type="range" min={50} max={95} value={Math.round(thresholdFinger * 100)} onChange={(e) => onThresholdFingerChange(Number(e.target.value) / 100)} />
    </div>
  );
}
