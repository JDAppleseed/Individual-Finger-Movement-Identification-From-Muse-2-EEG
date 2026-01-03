import React, { useMemo } from "react";
import { motion } from "framer-motion";
import { Play, Power, Square, SlidersHorizontal } from "lucide-react";
import clsx from "clsx";
import { shallow } from "zustand/shallow";
import { buildControlPayload, useControlMutation } from "../../api/client";
import type { ControlPayload } from "../../api/client";
import { useDemoStore } from "../../state/useDemoStore";

const sectionVariants = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.25 } }
};

export default function LeftRail() {
  const {
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
    setMcPasses,
    setSmoothingEnabled,
    setSmoothingMethod,
    setSmoothingWindow,
    setHysteresisEnabled,
    setHysteresisFrames,
    setThresholdAction,
    setThresholdFinger,
    setAdjacencyEnabled
  } = useDemoStore(
    (state) => ({
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
      setMcPasses: state.setMcPasses,
      setSmoothingEnabled: state.setSmoothingEnabled,
      setSmoothingMethod: state.setSmoothingMethod,
      setSmoothingWindow: state.setSmoothingWindow,
      setHysteresisEnabled: state.setHysteresisEnabled,
      setHysteresisFrames: state.setHysteresisFrames,
      setThresholdAction: state.setThresholdAction,
      setThresholdFinger: state.setThresholdFinger,
      setAdjacencyEnabled: state.setAdjacencyEnabled
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

  const sendControl = (overrides: Partial<ControlPayload> = {}) => {
    controlMutation.mutate(buildControlPayload(controlState, overrides));
  };

  return (
    <aside className="rail">
      <motion.div variants={sectionVariants} initial="hidden" animate="show" className="rail-panel">
        <div className="panel-title-row">
          <h3>Session Controls</h3>
          <span className="panel-tag">
            <Power size={14} /> Live
          </span>
        </div>

        <div className="segmented">
          <button
            className={clsx("segmented-btn", mode === "replay" && "active")}
            onClick={() => {
              setMode("replay");
              sendControl({ mode: "replay" });
            }}
          >
            Replay
          </button>
          <button
            className={clsx("segmented-btn", mode === "live" && "active")}
            onClick={() => {
              setMode("live");
              sendControl({ mode: "live" });
            }}
          >
            Live
          </button>
        </div>

        <div className="button-row">
          <button className="btn primary" onClick={() => sendControl({})}>
            <Play size={16} /> Start
          </button>
          <button
            className="btn"
            onClick={() => {
              setMode("idle");
              sendControl({ mode: "idle" });
            }}
          >
            <Square size={16} /> Stop
          </button>
        </div>

        <label className="control-label">Replay file</label>
        <input
          className="input"
          value={replayPath}
          onChange={(e) => {
            const next = e.target.value;
            setReplayPath(next);
            sendControl({ replay_path: next });
          }}
        />

        <label className="control-label">FPS: {fps}</label>
        <input
          className="slider"
          type="range"
          min={5}
          max={30}
          value={fps}
          onChange={(e) => {
            const next = Number(e.target.value);
            setFps(next);
            sendControl({ fps: next });
          }}
        />

        <label className="control-label">MC passes: {mcPasses}</label>
        <input
          className="slider"
          type="range"
          min={5}
          max={30}
          value={mcPasses}
          onChange={(e) => {
            const next = Number(e.target.value);
            setMcPasses(next);
            sendControl({ mc_passes: next });
          }}
        />
      </motion.div>

      <motion.div variants={sectionVariants} initial="hidden" animate="show" className="rail-panel">
        <div className="panel-title-row">
          <h3>Signal Conditioning</h3>
          <span className="panel-tag">
            <SlidersHorizontal size={14} /> Filters
          </span>
        </div>

        <div className="toggle-row">
          <label className="toggle">
            <input
              type="checkbox"
              checked={smoothingEnabled}
              onChange={(e) => {
                const next = e.target.checked;
                setSmoothingEnabled(next);
                sendControl({ smoothing_enabled: next });
              }}
            />
            Smoothing
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={hysteresisEnabled}
              onChange={(e) => {
                const next = e.target.checked;
                setHysteresisEnabled(next);
                sendControl({ hysteresis_enabled: next });
              }}
            />
            Hysteresis
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={adjacencyEnabled}
              onChange={(e) => {
                const next = e.target.checked;
                setAdjacencyEnabled(next);
                sendControl({ adjacency_enabled: next });
              }}
            />
            Adjacency assist
          </label>
        </div>

        <label className="control-label">Method</label>
        <select
          className="input"
          value={smoothingMethod}
          onChange={(e) => {
            const next = e.target.value;
            setSmoothingMethod(next);
            sendControl({ smoothing_method: next });
          }}
        >
          <option value="vote">Vote</option>
          <option value="ema">EMA</option>
        </select>

        <label className="control-label">Window N: {smoothingWindow}</label>
        <input
          className="slider"
          type="range"
          min={3}
          max={15}
          value={smoothingWindow}
          onChange={(e) => {
            const next = Number(e.target.value);
            setSmoothingWindow(next);
            sendControl({ smoothing_window: next });
          }}
        />

        <label className="control-label">Hysteresis frames: {hysteresisFrames}</label>
        <input
          className="slider"
          type="range"
          min={2}
          max={8}
          value={hysteresisFrames}
          onChange={(e) => {
            const next = Number(e.target.value);
            setHysteresisFrames(next);
            sendControl({ hysteresis_frames: next });
          }}
        />
      </motion.div>

      <motion.div variants={sectionVariants} initial="hidden" animate="show" className="rail-panel">
        <div className="panel-title-row">
          <h3>Decision Thresholds</h3>
          <span className="panel-tag">Confidence</span>
        </div>

        <label className="control-label">Action threshold: {Math.round(thresholdAction * 100)}%</label>
        <input
          className="slider"
          type="range"
          min={50}
          max={95}
          value={Math.round(thresholdAction * 100)}
          onChange={(e) => {
            const next = Number(e.target.value) / 100;
            setThresholdAction(next);
            sendControl({ threshold_action: next });
          }}
        />

        <label className="control-label">Finger threshold: {Math.round(thresholdFinger * 100)}%</label>
        <input
          className="slider"
          type="range"
          min={50}
          max={95}
          value={Math.round(thresholdFinger * 100)}
          onChange={(e) => {
            const next = Number(e.target.value) / 100;
            setThresholdFinger(next);
            sendControl({ threshold_finger: next });
          }}
        />
      </motion.div>
    </aside>
  );
}
