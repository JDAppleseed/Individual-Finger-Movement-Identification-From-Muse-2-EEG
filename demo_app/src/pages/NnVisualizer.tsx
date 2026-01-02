import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Graph3D from "../components/nnvis/Graph3D";
import Heatmap from "../components/nnvis/Heatmap";
import LinePlot from "../components/nnvis/LinePlot";
import BarChart from "../components/nnvis/BarChart";
import TopEdges from "../components/nnvis/TopEdges";
import ConvKernelGrid from "../components/nnvis/ConvKernelGrid";
import { decodePacked } from "../components/nnvis/utils";
import type { InferenceTick, NnvisActivation } from "../types/schema";
import type { NnvisManifest, NnvisWeights, NnvisTimelineManifest, OfflineSource } from "../types/nnvis";

const API_BASE = "http://127.0.0.1:8008";

type Props = {
  ws: WebSocket | null;
  tick: InferenceTick | null;
};

function useLocalStorageState<T>(key: string, initial: T): [T, (next: T) => void] {
  const [state, setState] = useState<T>(() => {
    const raw = localStorage.getItem(key);
    if (raw) {
      try {
        return JSON.parse(raw) as T;
      } catch {
        return initial;
      }
    }
    return initial;
  });

  const setValue = (next: T) => {
    setState(next);
    localStorage.setItem(key, JSON.stringify(next));
  };

  return [state, setValue];
}

function exportCanvasPNG(canvas: HTMLCanvasElement | null, filename: string) {
  if (!canvas) return;
  const link = document.createElement("a");
  link.href = canvas.toDataURL("image/png");
  link.download = filename;
  link.click();
}

async function fetchJson(url: string): Promise<any> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (err) {
    const message = `fetchJson ${url} status 0: ${String(err)}`;
    console.error(message);
    throw new Error(message);
  }
  const text = await res.text();
  if (!res.ok) {
    const message = `fetchJson ${url} status ${res.status}: ${text}`;
    console.error(message);
    throw new Error(message);
  }
  try {
    return JSON.parse(text);
  } catch (err) {
    const message = `fetchJson ${url} status ${res.status}: ${text}`;
    console.error(message);
    throw new Error(message);
  }
}

export default function NnVisualizer({ ws, tick }: Props) {
  const [mode, setMode] = useState<"online" | "offline">("online");
  const [manifest, setManifest] = useState<NnvisManifest | null>(null);
  const [weights, setWeights] = useState<NnvisWeights | null>(null);
  const [baseWeights, setBaseWeights] = useState<NnvisWeights | null>(null);
  const [activations, setActivations] = useState<NnvisActivation | null>(null);
  const [sources, setSources] = useState<OfflineSource[]>([]);
  const [sourcePath, setSourcePath] = useState<string>("");
  const [sampleIndex, setSampleIndex] = useState<number>(0);
  const [playing, setPlaying] = useState<boolean>(false);
  const [subscribe, setSubscribe] = useState<boolean>(false);
  const [rateHz, setRateHz] = useState<number>(5);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<NnvisTimelineManifest | null>(null);
  const [timelineIndex, setTimelineIndex] = useState<number>(0);
  const [now, setNow] = useState<number>(Date.now());
  const [healthOk, setHealthOk] = useState<boolean | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [manifestError, setManifestError] = useState<string | null>(null);
  const [weightsError, setWeightsError] = useState<string | null>(null);
  const [sourcesError, setSourcesError] = useState<string | null>(null);

  const [showArchitecture, setShowArchitecture] = useLocalStorageState("nnvis.showArchitecture", true);
  const [showWeights, setShowWeights] = useLocalStorageState("nnvis.showWeights", true);
  const [showActivations, setShowActivations] = useLocalStorageState("nnvis.showActivations", true);
  const [showTimeline, setShowTimeline] = useLocalStorageState("nnvis.showTimeline", true);
  const [showShapes, setShowShapes] = useLocalStorageState("nnvis.showShapes", true);
  const [showParams, setShowParams] = useLocalStorageState("nnvis.showParams", true);
  const [showMacs, setShowMacs] = useLocalStorageState("nnvis.showMacs", true);
  const [grouping, setGrouping] = useLocalStorageState("nnvis.grouping", true);

  const lastUpdateRef = useRef<number>(0);

  const loadBootstrap = useCallback(async () => {
    setHealthError(null);
    setManifestError(null);
    setWeightsError(null);
    setSourcesError(null);

    try {
      await fetchJson(`${API_BASE}/health`);
      setHealthOk(true);
    } catch (err) {
      setHealthOk(false);
      setHealthError(err instanceof Error ? err.message : String(err));
    }

    try {
      const data = await fetchJson(`${API_BASE}/nnvis/manifest`);
      setManifest(data);
    } catch (err) {
      setManifest(null);
      setManifestError(err instanceof Error ? err.message : String(err));
    }

    try {
      const data = await fetchJson(`${API_BASE}/nnvis/weights?quantize=1&downsample=1&topk=150`);
      setWeights(data);
      setBaseWeights(data);
    } catch (err) {
      setWeights(null);
      setWeightsError(err instanceof Error ? err.message : String(err));
    }

    try {
      const data = await fetchJson(`${API_BASE}/nnvis/offline/sources`);
      const list = data.sources as OfflineSource[];
      setSources(list);
      if (list.length > 0) {
        setSourcePath((prev) => prev || list[0].path);
      }
    } catch (err) {
      setSources([]);
      setSourcesError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void loadBootstrap();
  }, [loadBootstrap]);

  useEffect(() => {
    if (sourcePath) {
      setSampleIndex(0);
      setPlaying(false);
    }
  }, [sourcePath]);

  useEffect(() => {
    if (!manifest?.timeline.available || !manifest.timeline.manifest_url) {
      setTimeline(null);
      return;
    }
    fetch(`${API_BASE}${manifest.timeline.manifest_url}`)
      .then((res) => res.json())
      .then((data) => {
        setTimeline(data);
        if (data.steps?.length) {
          setTimelineIndex(Math.max(0, data.steps.length - 1));
        }
      })
      .catch(() => setTimeline(null));
  }, [manifest]);

  useEffect(() => {
    if (!timeline || !timeline.steps.length) return;
    const step = timeline.steps[timelineIndex];
    if (!step) return;
    fetch(`${API_BASE}/nnvis/timeline/weights?file=${encodeURIComponent(step.file)}`)
      .then((res) => res.json())
      .then((data) => setWeights(data))
      .catch(() => setWeights(baseWeights));
  }, [timeline, timelineIndex, baseWeights]);

  useEffect(() => {
    if (mode !== "offline" || !sourcePath) return;
    const params = new URLSearchParams({ source: sourcePath, index: String(sampleIndex) });
    fetch(`${API_BASE}/nnvis/offline/sample?${params.toString()}`)
      .then((res) => res.json())
      .then((data) => {
        setActivations(data);
        lastUpdateRef.current = Date.now();
      })
      .catch(() => setActivations(null));
  }, [mode, sourcePath, sampleIndex]);

  useEffect(() => {
    if (mode !== "online") return;
    const nnvis = tick?.nnvis ?? null;
    if (nnvis) {
      setActivations(nnvis);
      lastUpdateRef.current = Date.now();
    }
  }, [mode, tick]);

  useEffect(() => {
    if (!ws) return;
    const payload = { type: "nnvis_subscribe", enabled: subscribe, rate_hz: rateHz };
    const sendIfReady = () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
        return true;
      }
      return false;
    };
    if (sendIfReady()) return;
    const timer = window.setInterval(() => {
      if (sendIfReady()) {
        window.clearInterval(timer);
      }
    }, 300);
    return () => window.clearInterval(timer);
  }, [ws, subscribe, rateHz]);

  useEffect(() => {
    if (!playing || mode !== "offline") return;
    const source = sources.find((s) => s.path === sourcePath);
    const maxIndex = source ? Math.max(0, source.sample_count - 1) : 0;
    const timer = window.setInterval(() => {
      setSampleIndex((prev) => (prev >= maxIndex ? 0 : prev + 1));
    }, 200);
    return () => window.clearInterval(timer);
  }, [playing, mode, sources, sourcePath]);

  useEffect(() => {
    // Keep stale indicator updating even when stream is idle.
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, []);

  const stale = useMemo(() => {
    if (!lastUpdateRef.current) return false;
    return now - lastUpdateRef.current > 2000;
  }, [now, activations]);

  const fingerLabels = useMemo(() => {
    const map = manifest?.labels.finger ?? {};
    return Object.keys(map).sort((a, b) => Number(a) - Number(b)).map((key) => map[key]);
  }, [manifest]);

  const actionLabels = useMemo(() => {
    const map = manifest?.labels.action ?? {};
    return Object.keys(map).sort((a, b) => Number(a) - Number(b)).map((key) => map[key]);
  }, [manifest]);

  const errorMessages = [
    healthError && `Health: ${healthError}`,
    manifestError && `Manifest: ${manifestError}`,
    weightsError && `Weights: ${weightsError}`,
    sourcesError && `Sources: ${sourcesError}`,
  ].filter(Boolean) as string[];

  const healthLabel = healthOk === null ? "Unknown" : (healthOk ? "Reachable" : "Unavailable");

  const conv1 = useMemo(() => {
    const entry = weights?.conv.find((c) => c.id === "conv1");
    return entry ? decodePacked(entry.weights) : null;
  }, [weights]);

  const conv2 = useMemo(() => {
    const entry = weights?.conv.find((c) => c.id === "conv2");
    return entry ? decodePacked(entry.weights) : null;
  }, [weights]);

  const fingerHead = useMemo(() => {
    const entry = weights?.linear.find((c) => c.id === "finger_head");
    return entry ? decodePacked(entry.weights) : null;
  }, [weights]);

  const actionHead = useMemo(() => {
    const entry = weights?.linear.find((c) => c.id === "action_head");
    return entry ? decodePacked(entry.weights) : null;
  }, [weights]);

  const lstmIh = useMemo(() => (weights ? decodePacked(weights.lstm.weight_ih_l0) : null), [weights]);
  const lstmHh = useMemo(() => (weights ? decodePacked(weights.lstm.weight_hh_l0) : null), [weights]);
  const lstmBiasIh = useMemo(() => (weights ? decodePacked(weights.lstm.bias_ih_l0) : null), [weights]);
  const lstmBiasHh = useMemo(() => (weights ? decodePacked(weights.lstm.bias_hh_l0) : null), [weights]);

  const inputDecoded = useMemo(() => {
    if (!activations) return null;
    return decodePacked(activations.input.values);
  }, [activations]);
  const conv1Act = useMemo(() => activations ? decodePacked(activations.conv1.values) : null, [activations]);
  const conv2Act = useMemo(() => activations ? decodePacked(activations.conv2.values) : null, [activations]);
  const lstmAct = useMemo(() => activations ? decodePacked(activations.lstm_out.values) : null, [activations]);
  const lastFeat = useMemo(() => activations ? decodePacked(activations.last_features.values) : null, [activations]);

  const selectedNodeInfo = useMemo(() => manifest?.nodes.find((n) => n.id === selectedNode), [manifest, selectedNode]);

  const gridRef = useRef<HTMLCanvasElement | null>(null);
  const gridRef2 = useRef<HTMLCanvasElement | null>(null);
  const convInspectRef = useRef<HTMLCanvasElement | null>(null);
  const conv2InspectRef = useRef<HTMLCanvasElement | null>(null);
  const fingerRef = useRef<HTMLCanvasElement | null>(null);
  const actionRef = useRef<HTMLCanvasElement | null>(null);
  const lstmIhRef = useRef<HTMLCanvasElement | null>(null);
  const lstmHhRef = useRef<HTMLCanvasElement | null>(null);

  const [convInspectIndex, setConvInspectIndex] = useState<number>(0);
  const [convInspectLayer, setConvInspectLayer] = useState<"conv1" | "conv2">("conv1");

  useEffect(() => {
    const outCh = convInspectLayer === "conv1" ? conv1?.shape[0] : conv2?.shape[0];
    if (outCh !== undefined && convInspectIndex > outCh - 1) {
      setConvInspectIndex(0);
    }
  }, [convInspectLayer, conv1, conv2, convInspectIndex]);

  const convInspectData = useMemo(() => {
    const decoded = convInspectLayer === "conv1" ? conv1 : conv2;
    if (!decoded) return null;
    const shape = decoded.shape as number[];
    const [outCh, inCh, k] = shape as [number, number, number];
    const idx = Math.min(Math.max(convInspectIndex, 0), outCh - 1);
    const slice: number[] = [];
    for (let i = 0; i < inCh; i += 1) {
      for (let t = 0; t < k; t += 1) {
        const offset = idx * inCh * k + i * k + t;
        slice.push(decoded.data[offset] ?? 0);
      }
    }
    return { data: Float32Array.from(slice), shape: [inCh, k] as [number, number] };
  }, [convInspectLayer, convInspectIndex, conv1, conv2]);

  const topkIh = useMemo(() => weights?.lstm.topk.edges.filter((e) => e.matrix === "weight_ih_l0") ?? [], [weights]);
  const topkHh = useMemo(() => weights?.lstm.topk.edges.filter((e) => e.matrix === "weight_hh_l0") ?? [], [weights]);

  return (
    <div className="nnvis-main">
      <div className="nnvis">
      <div className="nnvis-header">
        <div>
          <h2>NN Visualizer</h2>
          <p>Inspect architecture, weights, and activations for the CNN+LSTM model.</p>
        </div>
        <div className="nnvis-controls">
          <div className="nnvis-toggle-row">
            <label>
              <input type="radio" checked={mode === "online"} onChange={() => setMode("online")} />
              Online
            </label>
            <label>
              <input type="radio" checked={mode === "offline"} onChange={() => setMode("offline")} />
              Offline
            </label>
          </div>
          <div className="nnvis-toggle-row">
            <label>
              <input type="checkbox" checked={showArchitecture} onChange={(e) => setShowArchitecture(e.target.checked)} />
              Architecture
            </label>
            <label>
              <input type="checkbox" checked={showWeights} onChange={(e) => setShowWeights(e.target.checked)} />
              Weights
            </label>
            <label>
              <input type="checkbox" checked={showActivations} onChange={(e) => setShowActivations(e.target.checked)} />
              Activations
            </label>
            <label>
              <input type="checkbox" checked={showTimeline} onChange={(e) => setShowTimeline(e.target.checked)} />
              Timeline
            </label>
          </div>
        </div>
      </div>

      <section className="nnvis-panel">
        <div className="nnvis-panel-title">Connection</div>
        <div className="nnvis-inline">
          <div className="nnvis-card">
            <div className="nnvis-card-title">Backend</div>
            <div className="nnvis-card-value">{healthLabel}</div>
          </div>
          <button className="nnvis-button" onClick={() => void loadBootstrap()}>Retry</button>
        </div>
        {errorMessages.length > 0 && (
          <div className="nnvis-warning">{errorMessages.join(" | ")}</div>
        )}
        <div className="nnvis-inline">
          {!manifest && (
            <div className="nnvis-card">
              <div className="nnvis-card-title">Waiting for manifest...</div>
            </div>
          )}
          {!weights && (
            <div className="nnvis-card">
              <div className="nnvis-card-title">Waiting for weights...</div>
            </div>
          )}
        </div>
      </section>

      <section className="nnvis-panel">
        <div className="nnvis-panel-title">Mode Controls</div>
        {mode === "online" ? (
          <div className="nnvis-control-grid">
            <label className="nnvis-switch">
              <input type="checkbox" checked={subscribe} onChange={(e) => setSubscribe(e.target.checked)} />
              Subscribe activations
            </label>
            <label className="nnvis-slider">
              Update rate (Hz)
              <input
                type="range"
                min={1}
                max={20}
                step={1}
                value={rateHz}
                onChange={(e) => setRateHz(Number(e.target.value))}
              />
              <span>{rateHz} Hz</span>
            </label>
          </div>
        ) : (
          <div className="nnvis-control-grid">
            <label className="nnvis-select">
              Offline source
              <select value={sourcePath} onChange={(e) => setSourcePath(e.target.value)}>
                {sources.map((s) => (
                  <option key={s.path} value={s.path}>
                    {s.name} ({s.sample_count})
                  </option>
                ))}
              </select>
            </label>
            <label className="nnvis-slider">
              Sample index
              <input
                type="range"
                min={0}
                max={Math.max(0, (sources.find((s) => s.path === sourcePath)?.sample_count ?? 1) - 1)}
                value={sampleIndex}
                onChange={(e) => setSampleIndex(Number(e.target.value))}
              />
              <span>{sampleIndex}</span>
            </label>
            <div className="nnvis-playback">
              <button className="nnvis-button" onClick={() => setSampleIndex((prev) => Math.max(prev - 1, 0))}>-</button>
              <button className="nnvis-button" onClick={() => setPlaying((prev) => !prev)}>{playing ? "Pause" : "Play"}</button>
              <button className="nnvis-button" onClick={() => setSampleIndex((prev) => prev + 1)}>+</button>
            </div>
          </div>
        )}
        {mode === "offline" && sources.length === 0 && (
          <div className="nnvis-warning">No offline npz sources found. Place `test_predictions.npz` or `eeg_windows.npz` in the repo.</div>
        )}
      </section>

      <section className="nnvis-panel">
        <div className="nnvis-panel-title">Diagnostics</div>
        <div className="nnvis-diagnostics">
          <div className="nnvis-card">
            <div className="nnvis-card-title">Prediction</div>
            <div className="nnvis-card-value">Action: {activations?.probs.action.pred_name ?? "--"}</div>
            <div className="nnvis-card-value">Finger: {activations?.probs.finger.pred_name ?? "--"}</div>
            <div className="nnvis-card-sub">Confidence: {activations ? Math.max(...activations.probs.action.values).toFixed(2) : "--"}</div>
          </div>
          <div className="nnvis-card">
            <div className="nnvis-card-title">Stream</div>
            <div className="nnvis-card-value">Mode: {mode.toUpperCase()}</div>
            <div className="nnvis-card-value">Sample: {activations?.sample.index ?? "--"}</div>
            <div className={`nnvis-card-sub ${stale ? "nnvis-stale" : ""}`}>{stale ? "Stale data" : "Live"}</div>
          </div>
          <div className="nnvis-card">
            <div className="nnvis-card-title">Uncertainty</div>
            <div className="nnvis-card-value">Entropy: {activations?.uncertainty.action_entropy?.toFixed(3) ?? "--"}</div>
            <div className="nnvis-card-value">MI: {activations?.uncertainty.action_mi?.toFixed(3) ?? "--"}</div>
            <div className="nnvis-card-sub">Std mean: {activations?.uncertainty.action_std_mean?.toFixed(3) ?? "--"}</div>
          </div>
        </div>
      </section>

      {showArchitecture && manifest && (
        <section className="nnvis-panel">
          <div className="nnvis-panel-title">Architecture</div>
          <div className="nnvis-architecture">
            <Graph3D
              nodes={manifest.nodes}
              edges={manifest.edges}
              grouping={grouping}
              selectedId={selectedNode}
              onSelect={setSelectedNode}
            />
            <div className="nnvis-node-details">
              <div className="nnvis-toggle-row">
                <label>
                  <input type="checkbox" checked={showShapes} onChange={(e) => setShowShapes(e.target.checked)} />
                  Shapes
                </label>
                <label>
                  <input type="checkbox" checked={showParams} onChange={(e) => setShowParams(e.target.checked)} />
                  Params
                </label>
                <label>
                  <input type="checkbox" checked={showMacs} onChange={(e) => setShowMacs(e.target.checked)} />
                  MACs
                </label>
                <label>
                  <input type="checkbox" checked={grouping} onChange={(e) => setGrouping(e.target.checked)} />
                  Grouping
                </label>
              </div>
              <div className="nnvis-card">
                <div className="nnvis-card-title">Selected Node</div>
                {selectedNodeInfo ? (
                  <>
                    <div className="nnvis-card-value">{selectedNodeInfo.title}</div>
                    {showShapes && (
                      <div className="nnvis-card-sub">
                        {selectedNodeInfo.shape ?? `${selectedNodeInfo.shape_in ?? ""} → ${selectedNodeInfo.shape_out ?? ""}`}
                      </div>
                    )}
                    {showParams && <div className="nnvis-card-sub">Params: {selectedNodeInfo.params}</div>}
                    {showMacs && <div className="nnvis-card-sub">MACs: {selectedNodeInfo.macs}</div>}
                  </>
                ) : (
                  <div className="nnvis-card-sub">Click a node to inspect.</div>
                )}
              </div>
              <div className="nnvis-card">
                <div className="nnvis-card-title">Totals</div>
                <div className="nnvis-card-value">Params: {manifest.totals.params}</div>
                <div className="nnvis-card-value">MACs/window: {manifest.totals.macs_per_window}</div>
              </div>
            </div>
          </div>
        </section>
      )}

      {showWeights && weights && (
        <section className="nnvis-panel">
          <div className="nnvis-panel-title">Weights</div>
          <div className="nnvis-weights">
            <div className="nnvis-card">
              <div className="nnvis-card-title">Conv1 Kernels Grid</div>
              {conv1 && (
                <>
                  <ConvKernelGrid ref={gridRef} data={conv1.data} shape={conv1.shape as [number, number, number]} />
                  <button className="nnvis-button" onClick={() => exportCanvasPNG(gridRef.current, "conv1_grid.png")}>Export PNG</button>
                </>
              )}
            </div>

            <div className="nnvis-card">
              <div className="nnvis-card-title">Conv2 Kernels Grid</div>
              {conv2 && (
                <>
                  <ConvKernelGrid ref={gridRef2} data={conv2.data} shape={conv2.shape as [number, number, number]} />
                  <button className="nnvis-button" onClick={() => exportCanvasPNG(gridRef2.current, "conv2_grid.png")}>Export PNG</button>
                </>
              )}
            </div>

            <div className="nnvis-card">
              <div className="nnvis-card-title">Conv Filter Inspect</div>
              <div className="nnvis-inline">
                <select value={convInspectLayer} onChange={(e) => setConvInspectLayer(e.target.value as "conv1" | "conv2")}>
                  <option value="conv1">conv1</option>
                  <option value="conv2">conv2</option>
                </select>
                <input
                  type="number"
                  min={0}
                  max={Math.max(0, ((convInspectLayer === "conv1" ? conv1?.shape[0] : conv2?.shape[0]) ?? 1) - 1)}
                  value={convInspectIndex}
                  onChange={(e) => setConvInspectIndex(Number(e.target.value))}
                />
              </div>
              {convInspectData && (
                <>
                  <Heatmap ref={convInspectLayer === "conv1" ? convInspectRef : conv2InspectRef} data={convInspectData.data} shape={convInspectData.shape} />
                  <button
                    className="nnvis-button"
                    onClick={() => exportCanvasPNG(convInspectLayer === "conv1" ? convInspectRef.current : conv2InspectRef.current, "conv_filter.png")}
                  >
                    Export PNG
                  </button>
                </>
              )}
            </div>

            <div className="nnvis-card">
              <div className="nnvis-card-title">Finger Head</div>
              {fingerHead && (
                <>
                  <Heatmap ref={fingerRef} data={fingerHead.data} shape={fingerHead.shape as [number, number]} />
                  <button className="nnvis-button" onClick={() => exportCanvasPNG(fingerRef.current, "finger_head.png")}>Export PNG</button>
                </>
              )}
            </div>

            <div className="nnvis-card">
              <div className="nnvis-card-title">Action Head</div>
              {actionHead && (
                <>
                  <Heatmap ref={actionRef} data={actionHead.data} shape={actionHead.shape as [number, number]} />
                  <button className="nnvis-button" onClick={() => exportCanvasPNG(actionRef.current, "action_head.png")}>Export PNG</button>
                </>
              )}
            </div>

            <div className="nnvis-card">
              <div className="nnvis-card-title">LSTM weight_ih_l0</div>
              {lstmIh && (
                <>
                  <Heatmap
                    ref={lstmIhRef}
                    data={lstmIh.data}
                    shape={lstmIh.shape as [number, number]}
                    highlights={topkIh.slice(0, 80).map((e) => ({ row: e.i, col: e.j }))}
                  />
                  <button className="nnvis-button" onClick={() => exportCanvasPNG(lstmIhRef.current, "lstm_weight_ih.png")}>Export PNG</button>
                  <TopEdges edges={topkIh.slice(0, 120)} shape={lstmIh.shape as [number, number]} />
                </>
              )}
            </div>

            <div className="nnvis-card">
              <div className="nnvis-card-title">LSTM weight_hh_l0</div>
              {lstmHh && (
                <>
                  <Heatmap
                    ref={lstmHhRef}
                    data={lstmHh.data}
                    shape={lstmHh.shape as [number, number]}
                    highlights={topkHh.slice(0, 80).map((e) => ({ row: e.i, col: e.j }))}
                  />
                  <button className="nnvis-button" onClick={() => exportCanvasPNG(lstmHhRef.current, "lstm_weight_hh.png")}>Export PNG</button>
                  <TopEdges edges={topkHh.slice(0, 120)} shape={lstmHh.shape as [number, number]} />
                </>
              )}
            </div>

            <div className="nnvis-card">
              <div className="nnvis-card-title">LSTM Bias</div>
              <div className="nnvis-inline">
                {lstmBiasIh && <BarChart data={Array.from(lstmBiasIh.data)} width={260} height={120} />}
                {lstmBiasHh && <BarChart data={Array.from(lstmBiasHh.data)} width={260} height={120} />}
              </div>
            </div>
          </div>
        </section>
      )}

      {!activations && (
        <section className="nnvis-panel">
          <div className="nnvis-panel-title">No activations yet</div>
          <div className="nnvis-card">
            <div className="nnvis-card-title">
              Waiting for stream (ONLINE) or load a sample (OFFLINE).
            </div>
            <div className="nnvis-card-value">
              {stale ? "Stale data" : "No data received yet"}
            </div>
          </div>
        </section>
      )}

      {showActivations && (
        <section className="nnvis-panel">
          <div className="nnvis-panel-title">Activations</div>
          {!activations && <div className="nnvis-warning">No activations available. Enable subscription or select offline sample.</div>}
          {activations && (
            <div className="nnvis-activations">
              <div className="nnvis-card">
                <div className="nnvis-card-title">Input Signals</div>
                {inputDecoded && (
                  <LinePlot
                    data={inputDecoded.data}
                    shape={[manifest?.input.timesteps ?? 64, manifest?.input.channels ?? 4]}
                    labels={manifest?.input.channel_names}
                  />
                )}
              </div>
              <div className="nnvis-card">
                <div className="nnvis-card-title">Conv1 Activations</div>
                {conv1Act && <Heatmap data={conv1Act.data} shape={[16, 64]} />}
              </div>
              <div className="nnvis-card">
                <div className="nnvis-card-title">Conv2 Activations</div>
                {conv2Act && <Heatmap data={conv2Act.data} shape={[32, 64]} />}
              </div>
              <div className="nnvis-card">
                <div className="nnvis-card-title">LSTM Outputs</div>
                {lstmAct && <Heatmap data={lstmAct.data} shape={[64, 64]} />}
              </div>
              <div className="nnvis-card">
                <div className="nnvis-card-title">Last Timestep Features</div>
                {lastFeat && <BarChart data={Array.from(lastFeat.data)} width={360} height={140} />}
              </div>
              <div className="nnvis-card">
                <div className="nnvis-card-title">Probabilities</div>
                <div className="nnvis-inline">
                  <BarChart
                    data={activations.probs.finger.values}
                    labels={fingerLabels}
                    highlightIndex={activations.probs.finger.pred_id}
                  />
                  <BarChart
                    data={activations.probs.action.values}
                    labels={actionLabels}
                    highlightIndex={activations.probs.action.pred_id}
                  />
                </div>
              </div>
            </div>
          )}
        </section>
      )}

      {showTimeline && (
        <section className="nnvis-panel">
          <div className="nnvis-panel-title">Timeline</div>
          {manifest?.timeline.available && timeline ? (
            <div className="nnvis-timeline">
              <input
                type="range"
                min={0}
                max={Math.max(0, timeline.steps.length - 1)}
                value={timelineIndex}
                onChange={(e) => setTimelineIndex(Number(e.target.value))}
              />
              <div className="nnvis-card-sub">
                {timeline.steps[timelineIndex]?.label ?? "Step"}
              </div>
              <button className="nnvis-button" onClick={() => setWeights(baseWeights)}>Use current weights</button>
            </div>
          ) : (
            <div className="nnvis-warning">No timeline exports found.</div>
          )}
        </section>
      )}
      </div>
    </div>
  );
}
