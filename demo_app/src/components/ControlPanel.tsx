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
  onMcPassesChange
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
    </div>
  );
}
