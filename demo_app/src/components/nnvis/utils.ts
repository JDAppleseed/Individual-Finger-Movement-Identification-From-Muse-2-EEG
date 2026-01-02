export type PackedEncoding = {
  encoding: "f16_base64";
  shape: number[];
  data: string;
};

export type PackedArray = number[] | number[][] | number[][][] | PackedEncoding;

function inferShape(values: any): number[] {
  if (!Array.isArray(values)) {
    return [];
  }
  if (values.length === 0) {
    return [0];
  }
  return [values.length, ...inferShape(values[0])];
}

function flatten(values: any, out: number[]): void {
  if (Array.isArray(values)) {
    for (const item of values) {
      flatten(item, out);
    }
    return;
  }
  out.push(typeof values === "number" ? values : Number(values));
}

function float16ToFloat32(value: number): number {
  const s = (value & 0x8000) >> 15;
  const e = (value & 0x7c00) >> 10;
  const f = value & 0x03ff;
  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
  }
  if (e === 0x1f) {
    return f ? NaN : (s ? -Infinity : Infinity);
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / Math.pow(2, 10));
}

function decodeBase64F16(payload: PackedEncoding): Float32Array {
  const binary = atob(payload.data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  const view = new Uint16Array(bytes.buffer);
  const out = new Float32Array(view.length);
  for (let i = 0; i < view.length; i += 1) {
    out[i] = float16ToFloat32(view[i]);
  }
  return out;
}

export function decodePacked(values: PackedArray): { data: Float32Array; shape: number[] } {
  if (typeof values === "object" && values !== null && !Array.isArray(values) && "encoding" in values) {
    const payload = values as PackedEncoding;
    return { data: decodeBase64F16(payload), shape: payload.shape };
  }
  const shape = inferShape(values);
  const flat: number[] = [];
  flatten(values, flat);
  return { data: Float32Array.from(flat), shape };
}

export function valueToColor(value: number, maxAbs: number): string {
  if (maxAbs <= 0) {
    return "rgb(20, 24, 32)";
  }
  const v = Math.max(-maxAbs, Math.min(maxAbs, value)) / maxAbs;
  const r = v > 0 ? 255 : Math.round(40 + 120 * (1 + v));
  const b = v < 0 ? 255 : Math.round(40 + 120 * (1 - v));
  const g = Math.round(60 + 120 * (1 - Math.abs(v)));
  return `rgb(${r}, ${g}, ${b})`;
}

export function computeMaxAbs(data: Float32Array): number {
  let max = 0;
  for (let i = 0; i < data.length; i += 1) {
    const v = Math.abs(data[i]);
    if (v > max) max = v;
  }
  return max;
}
