export type NnvisManifest = {
  model_name: string;
  input: { timesteps: number; channels: number; channel_names: string[] };
  labels: {
    action: Record<string, string>;
    finger: Record<string, string>;
  };
  nodes: {
    id: string;
    title: string;
    kind: string;
    shape?: string;
    shape_in?: string;
    shape_out?: string;
    params: number;
    macs: number;
  }[];
  edges: { from: string; to: string }[];
  totals: { params: number; macs_per_window: number };
  timeline: { available: boolean; manifest_url?: string };
};

export type PackedEncoding = {
  encoding: "f16_base64";
  shape: number[];
  data: string;
};

export type PackedArray = number[] | number[][] | number[][][] | PackedEncoding;

export type NnvisWeights = {
  version: number;
  conv: {
    id: string;
    weight_shape: number[];
    weights: PackedArray;
    bias: PackedArray | null;
  }[];
  linear: {
    id: string;
    weight_shape: number[];
    weights: PackedArray;
    bias: PackedArray;
  }[];
  lstm: {
    id: string;
    weight_ih_l0_shape: number[];
    weight_hh_l0_shape: number[];
    bias_ih_l0_shape: number[];
    bias_hh_l0_shape: number[];
    weight_ih_l0: PackedArray;
    weight_hh_l0: PackedArray;
    bias_ih_l0: PackedArray;
    bias_hh_l0: PackedArray;
    topk: { k: number; edges: { matrix: string; i: number; j: number; v: number }[] };
  };
};

export type NnvisTimelineManifest = {
  steps: { step: number; label: string; file: string }[];
};

export type OfflineSource = {
  path: string;
  name: string;
  sample_count: number;
};
