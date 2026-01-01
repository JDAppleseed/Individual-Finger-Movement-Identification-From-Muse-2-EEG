import { z } from "zod";

export const InferenceTickSchema = z.object({
  type: z.literal("tick"),
  ts_utc: z.string(),
  mode: z.enum(["replay", "live", "idle"]),
  session: z.object({
    subject_id: z.string(),
    experiment_hash: z.string(),
    window_index: z.number().int(),
    window_start_s: z.number(),
    window_end_s: z.number()
  }),
  prediction: z.object({
    action_id: z.number().int(),
    action_name: z.string(),
    finger_id: z.number().int(),
    finger_name: z.string(),
    action_confidence: z.number(),
    action_uncertainty: z.number(),
    finger_confidence: z.number(),
    finger_uncertainty: z.number()
  }),
  safety: z.object({
    base_threshold: z.number(),
    adaptive_threshold: z.number(),
    allow_actuation: z.boolean(),
    stability_frames: z.number().int(),
    stability_ok: z.boolean(),
    velocity: z.number()
  }),
  diagnostics: z.object({
    latency_ms: z.number(),
    fps_target: z.number(),
    fps_actual: z.number(),
    health_score: z.number(),
    lsl_connected: z.boolean(),
    artifact_suppression: z.boolean().nullable(),
    notes: z.string(),
    smoothing_enabled: z.boolean().optional(),
    smoothing_method: z.string().optional(),
    smoothing_window: z.number().optional(),
    hysteresis_enabled: z.boolean().optional(),
    hysteresis_frames: z.number().optional(),
    threshold_action: z.number().optional(),
    threshold_finger: z.number().optional(),
    adjacency_enabled: z.boolean().optional(),
    decision_reason: z.string().optional(),
    raw_top_action_id: z.number().int().optional(),
    raw_top_finger_id: z.number().int().optional(),
    committed_action_id: z.number().int().optional(),
    committed_finger_id: z.number().int().optional(),
    smoothed_action_id: z.number().int().optional(),
    smoothed_finger_id: z.number().int().optional(),
    frames_in_state: z.number().int().optional()
  })
});

export const StatusSchema = z.object({
  type: z.literal("status"),
  ts_utc: z.string(),
  level: z.enum(["info", "warning", "error"]),
  message: z.string(),
  details: z.record(z.any())
});

export type InferenceTick = z.infer<typeof InferenceTickSchema>;
export type StatusMessage = z.infer<typeof StatusSchema>;

export function parseMessage(payload: unknown): InferenceTick | StatusMessage | null {
  const tick = InferenceTickSchema.safeParse(payload);
  if (tick.success) {
    return tick.data;
  }
  const status = StatusSchema.safeParse(payload);
  if (status.success) {
    return status.data;
  }
  return null;
}
