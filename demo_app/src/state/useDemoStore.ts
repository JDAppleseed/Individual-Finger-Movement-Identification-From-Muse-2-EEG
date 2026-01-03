import { create } from "zustand";
import type { InferenceTick, StatusMessage } from "../types/schema";
import type { NnvisManifest } from "../types/nnvis";

export type WsState = "connecting" | "open" | "closed" | "error";

export type SelectedNodeMeta = NnvisManifest["nodes"][number];

export type DemoEvent = {
  id: string;
  ts: number;
  level: "info" | "warning" | "error";
  message: string;
};

type DemoState = {
  backendReachable: boolean | null;
  wsState: WsState;
  lastWsMessageAt: number | null;
  ws: WebSocket | null;

  stage: "nn" | "hand";

  mode: "replay" | "live" | "idle";
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

  tick: InferenceTick | null;
  status: StatusMessage | null;
  timeline: string[];
  selectedNodeId: string | null;
  selectedNodeMeta: SelectedNodeMeta | null;
  hoveredNodeId: string | null;
  hoveredNodeMeta: SelectedNodeMeta | null;
  eventLog: DemoEvent[];

  setBackendReachable: (value: boolean | null) => void;
  setWsState: (state: WsState) => void;
  setLastWsMessageAt: (value: number | null) => void;
  setWs: (ws: WebSocket | null) => void;

  setStage: (stage: "nn" | "hand") => void;

  setMode: (mode: "replay" | "live" | "idle") => void;
  setReplayPath: (value: string) => void;
  setFps: (value: number) => void;
  setMcPasses: (value: number) => void;
  setSmoothingEnabled: (value: boolean) => void;
  setSmoothingMethod: (value: string) => void;
  setSmoothingWindow: (value: number) => void;
  setHysteresisEnabled: (value: boolean) => void;
  setHysteresisFrames: (value: number) => void;
  setThresholdAction: (value: number) => void;
  setThresholdFinger: (value: number) => void;
  setAdjacencyEnabled: (value: boolean) => void;

  setTick: (tick: InferenceTick | null) => void;
  setStatus: (status: StatusMessage | null) => void;
  appendTimeline: (action: string) => void;
  setSelectedNode: (id: string | null, meta?: SelectedNodeMeta | null) => void;
  setHoveredNode: (id: string | null, meta?: SelectedNodeMeta | null) => void;
  pushEvent: (event: Omit<DemoEvent, "id" | "ts">) => void;
  clearEvents: () => void;
};

const MAX_TIMELINE = 120;
const MAX_EVENTS = 120;

export const useDemoStore = create<DemoState>((set) => ({
  backendReachable: null,
  wsState: "closed",
  lastWsMessageAt: null,
  ws: null,

  stage: "nn",

  mode: "replay",
  replayPath: "eeg_windows.npz",
  fps: 20,
  mcPasses: 10,
  smoothingEnabled: true,
  smoothingMethod: "vote",
  smoothingWindow: 5,
  hysteresisEnabled: true,
  hysteresisFrames: 3,
  thresholdAction: 0.75,
  thresholdFinger: 0.75,
  adjacencyEnabled: true,

  tick: null,
  status: null,
  timeline: [],
  selectedNodeId: null,
  selectedNodeMeta: null,
  hoveredNodeId: null,
  hoveredNodeMeta: null,
  eventLog: [],

  setBackendReachable: (value) => set({ backendReachable: value }),
  setWsState: (state) => set({ wsState: state }),
  setLastWsMessageAt: (value) => set({ lastWsMessageAt: value }),
  setWs: (ws) => set({ ws }),

  setStage: (stage) => set({ stage }),

  setMode: (mode) => set({ mode }),
  setReplayPath: (value) => set({ replayPath: value }),
  setFps: (value) => set({ fps: value }),
  setMcPasses: (value) => set({ mcPasses: value }),
  setSmoothingEnabled: (value) => set({ smoothingEnabled: value }),
  setSmoothingMethod: (value) => set({ smoothingMethod: value }),
  setSmoothingWindow: (value) => set({ smoothingWindow: value }),
  setHysteresisEnabled: (value) => set({ hysteresisEnabled: value }),
  setHysteresisFrames: (value) => set({ hysteresisFrames: value }),
  setThresholdAction: (value) => set({ thresholdAction: value }),
  setThresholdFinger: (value) => set({ thresholdFinger: value }),
  setAdjacencyEnabled: (value) => set({ adjacencyEnabled: value }),

  setTick: (tick) => set({ tick }),
  setStatus: (status) => set({ status }),
  appendTimeline: (action) =>
    set((state) => {
      const next = [...state.timeline, action].slice(-MAX_TIMELINE);
      return { timeline: next };
    }),
  setSelectedNode: (id, meta) => set({ selectedNodeId: id, selectedNodeMeta: meta ?? null }),
  setHoveredNode: (id, meta) => set({ hoveredNodeId: id, hoveredNodeMeta: meta ?? null }),
  pushEvent: (event) =>
    set((state) => {
      const entry: DemoEvent = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        ts: Date.now(),
        level: event.level,
        message: event.message
      };
      const next = [...state.eventLog, entry].slice(-MAX_EVENTS);
      return { eventLog: next };
    }),
  clearEvents: () => set({ eventLog: [] })
}));
