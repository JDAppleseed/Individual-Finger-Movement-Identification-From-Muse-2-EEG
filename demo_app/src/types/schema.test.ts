import { describe, expect, it } from "vitest";
import { InferenceTickSchema } from "./schema";

const sampleTick = {
  type: "tick",
  ts_utc: "2025-12-29T22:10:05.123Z",
  mode: "replay",
  session: {
    subject_id: "1-M60",
    experiment_hash: "abc123def456",
    window_index: 1234,
    window_start_s: 12.3,
    window_end_s: 12.55
  },
  prediction: {
    action_id: 1,
    action_name: "OPEN",
    finger_id: 2,
    finger_name: "INDEX",
    action_confidence: 0.82,
    action_uncertainty: 0.05,
    finger_confidence: 0.77,
    finger_uncertainty: 0.08
  },
  safety: {
    base_threshold: 0.75,
    adaptive_threshold: 0.79,
    allow_actuation: true,
    stability_frames: 3,
    stability_ok: true,
    velocity: 0.78
  },
  diagnostics: {
    latency_ms: 12.4,
    fps_target: 20,
    fps_actual: 19.6,
    health_score: 0.91,
    lsl_connected: false,
    artifact_suppression: null,
    notes: ""
  }
};

describe("schema", () => {
  it("parses tick payload", () => {
    const parsed = InferenceTickSchema.safeParse(sampleTick);
    expect(parsed.success).toBe(true);
  });
});
