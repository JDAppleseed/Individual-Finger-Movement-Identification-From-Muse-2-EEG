import React from "react";
import { motion } from "framer-motion";
import { Activity, AlertTriangle, CheckCircle2, Fingerprint, Gauge, Info, Shield } from "lucide-react";
import { shallow } from "zustand/shallow";
import { ActionBadge } from "../ActionBadge";
import { ConfidenceMeter } from "../ConfidenceMeter";
import { TimelineStrip } from "../TimelineStrip";
import { useDemoStore } from "../../state/useDemoStore";

const sectionVariants = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0, transition: { duration: 0.25 } }
};

function formatTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function RightInspector() {
  const {
    tick,
    status,
    timeline,
    selectedNodeMeta,
    hoveredNodeMeta,
    eventLog,
    setSelectedNode
  } = useDemoStore(
    (state) => ({
      tick: state.tick,
      status: state.status,
      timeline: state.timeline,
      selectedNodeMeta: state.selectedNodeMeta,
      hoveredNodeMeta: state.hoveredNodeMeta,
      eventLog: state.eventLog,
      setSelectedNode: state.setSelectedNode
    }),
    shallow
  );

  const action = tick?.prediction.action_name ?? "REST";
  const finger = tick?.prediction.finger_name ?? "NONE";
  const actionConf = tick?.prediction.action_confidence ?? 0;
  const fingerConf = tick?.prediction.finger_confidence ?? 0;
  const latency = tick?.diagnostics.latency_ms ?? 0;
  const fpsActual = tick?.diagnostics.fps_actual ?? 0;
  const fpsTarget = tick?.diagnostics.fps_target ?? 0;
  const healthScore = tick?.diagnostics.health_score ?? 0;
  const allowActuation = tick?.safety.allow_actuation ?? false;

  const decisionReason = tick?.diagnostics.decision_reason ?? "--";

  return (
    <aside className="inspector">
      <motion.div variants={sectionVariants} initial="hidden" animate="show" className="rail-panel">
        <div className="panel-title-row">
          <h3>Tick Summary</h3>
          <span className={allowActuation ? "status-pill ok" : "status-pill warn"}>
            <Shield size={14} /> {allowActuation ? "Actuate" : "Blocked"}
          </span>
        </div>

        <ActionBadge label={action} tone={action === "OPEN" ? "open" : action === "CLOSE" ? "close" : "rest"} />
        <div className="inline-meta">
          <Fingerprint size={16} /> Finger: <strong>{finger}</strong>
        </div>

        <ConfidenceMeter label="Action Confidence" value={actionConf} />
        <ConfidenceMeter label="Finger Confidence" value={fingerConf} />

        <div className="stat-grid">
          <div className="stat-card">
            <Gauge size={16} />
            <div>
              <div className="stat-label">Latency</div>
              <div className="stat-value">{latency.toFixed(1)} ms</div>
            </div>
          </div>
          <div className="stat-card">
            <Activity size={16} />
            <div>
              <div className="stat-label">FPS</div>
              <div className="stat-value">
                {fpsActual.toFixed(1)} / {fpsTarget.toFixed(0)}
              </div>
            </div>
          </div>
          <div className="stat-card">
            <CheckCircle2 size={16} />
            <div>
              <div className="stat-label">Health</div>
              <div className="stat-value">{Math.round(healthScore * 100)}%</div>
            </div>
          </div>
        </div>

        <div className="inspector-note">
          <Info size={14} /> {decisionReason}
        </div>
      </motion.div>

      <motion.div variants={sectionVariants} initial="hidden" animate="show" className="rail-panel">
        <div className="panel-title-row">
          <h3>{selectedNodeMeta ? "Selected Node" : hoveredNodeMeta ? "Hover Preview" : "Selected Node"}</h3>
          {selectedNodeMeta && (
            <button className="btn" type="button" onClick={() => setSelectedNode(null, null)}>
              Clear selection
            </button>
          )}
        </div>
        {selectedNodeMeta || hoveredNodeMeta ? (
          <div className="node-details">
            <div className="node-title">{(selectedNodeMeta ?? hoveredNodeMeta)?.title}</div>
            <div className="node-meta">ID: {(selectedNodeMeta ?? hoveredNodeMeta)?.id}</div>
            <div className="node-meta">Kind: {(selectedNodeMeta ?? hoveredNodeMeta)?.kind}</div>
            <div className="node-meta">
              Shape: {(selectedNodeMeta ?? hoveredNodeMeta)?.shape ?? `${(selectedNodeMeta ?? hoveredNodeMeta)?.shape_in ?? ""} → ${(selectedNodeMeta ?? hoveredNodeMeta)?.shape_out ?? ""}`}
            </div>
            <div className="node-meta">Params: {(selectedNodeMeta ?? hoveredNodeMeta)?.params}</div>
            <div className="node-meta">MACs: {(selectedNodeMeta ?? hoveredNodeMeta)?.macs}</div>
          </div>
        ) : (
          <div className="empty-state">Hover or click a node in the NN stage.</div>
        )}
      </motion.div>

      <motion.div variants={sectionVariants} initial="hidden" animate="show" className="rail-panel">
        <div className="panel-title-row">
          <h3>Timeline</h3>
        </div>
        <TimelineStrip actions={timeline} />
        <div className="status-text">{status?.message ?? "Waiting for status..."}</div>
      </motion.div>

      <motion.div variants={sectionVariants} initial="hidden" animate="show" className="rail-panel">
        <div className="panel-title-row">
          <h3>Event Log</h3>
        </div>
        <div className="event-log">
          {eventLog.length === 0 ? (
            <div className="empty-state">No events yet.</div>
          ) : (
            eventLog
              .slice()
              .reverse()
              .slice(0, 8)
              .map((entry) => (
                <div key={entry.id} className={`event-row ${entry.level}`}>
                  {entry.level === "error" ? <AlertTriangle size={14} /> : <Info size={14} />}
                  <span className="event-time">{formatTime(entry.ts)}</span>
                  <span className="event-message">{entry.message}</span>
                </div>
              ))
          )}
        </div>
      </motion.div>
    </aside>
  );
}
