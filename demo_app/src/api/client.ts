import { useMutation, useQuery } from "@tanstack/react-query";

export const BACKEND_PORT = 8008;

export type ControlState = {
  mode: string;
  replayPath: string;
  fps: number;
  mcPasses: number;
  smoothingEnabled: boolean;
  smoothingMethod: string;
  smoothingWindow: number;
  hysteresisEnabled: boolean;
  hysteresisFrames: number;
  thresholdAction: number;
  thresholdFinger: number;
  adjacencyEnabled: boolean;
};

export type ControlPayload = {
  mode: string;
  replay_path: string;
  fps: number;
  device: string;
  mc_passes: number;
  smoothing_enabled: boolean;
  smoothing_method: string;
  smoothing_window: number;
  hysteresis_enabled: boolean;
  hysteresis_frames: number;
  threshold_action: number;
  threshold_finger: number;
  adjacency_enabled: boolean;
};

export function resolveHost(): string {
  const h = window.location.hostname;
  return !h || h === "localhost" ? "127.0.0.1" : h;
}

export function getBackendBase(): string {
  return `http://${resolveHost()}:${BACKEND_PORT}`;
}

export function getControlUrl(): string {
  return `${getBackendBase()}/control`;
}

export function getWsUrl(): string {
  return `ws://${resolveHost()}:${BACKEND_PORT}/stream`;
}

export function buildControlPayload(state: ControlState, overrides: Partial<ControlPayload> = {}): ControlPayload {
  return {
    mode: state.mode,
    replay_path: state.replayPath,
    fps: state.fps,
    device: "cpu",
    mc_passes: state.mcPasses,
    smoothing_enabled: state.smoothingEnabled,
    smoothing_method: state.smoothingMethod,
    smoothing_window: state.smoothingWindow,
    hysteresis_enabled: state.hysteresisEnabled,
    hysteresis_frames: state.hysteresisFrames,
    threshold_action: state.thresholdAction,
    threshold_finger: state.thresholdFinger,
    adjacency_enabled: state.adjacencyEnabled,
    ...overrides
  };
}

export async function fetchJson<T = unknown>(url: string, init?: RequestInit): Promise<T> {
  const fullUrl = url.startsWith("http") ? url : `${getBackendBase()}${url}`;
  let res: Response;
  try {
    res = await fetch(fullUrl, init);
  } catch (err) {
    const message = `fetchJson ${fullUrl} status 0: ${String(err)}`;
    console.error(message);
    throw new Error(message);
  }
  const text = await res.text();
  if (!res.ok) {
    const message = `fetchJson ${fullUrl} status ${res.status}: ${text}`;
    console.error(message);
    throw new Error(message);
  }
  try {
    return JSON.parse(text) as T;
  } catch (err) {
    const message = `fetchJson ${fullUrl} status ${res.status}: ${text}`;
    console.error(message);
    throw new Error(message);
  }
}

export async function postControl(payload: ControlPayload): Promise<void> {
  const res = await fetch(getControlUrl(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`control ${res.status}: ${text}`);
  }
}

export function useBackendHealthQuery() {
  return useQuery({
    queryKey: ["backend", "health", getBackendBase()],
    queryFn: () => fetchJson("/health"),
    refetchInterval: 4000,
    staleTime: 3500,
    retry: 1
  });
}

export function useControlMutation() {
  return useMutation({
    mutationFn: (payload: ControlPayload) => postControl(payload)
  });
}
