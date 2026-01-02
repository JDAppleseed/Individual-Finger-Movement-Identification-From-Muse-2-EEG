import React, { useEffect, useRef } from "react";

const COLORS = ["#38bdf8", "#f97316", "#a855f7", "#22c55e", "#eab308"];

export type LinePlotProps = {
  data: Float32Array;
  shape: [number, number];
  width?: number;
  height?: number;
  labels?: string[];
};

export default function LinePlot({ data, shape, width = 320, height = 140, labels }: LinePlotProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = width;
    canvas.height = height;
    const [timesteps, channels] = shape;
    ctx.clearRect(0, 0, width, height);
    const stride = channels;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < data.length; i += 1) {
      const v = data[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;

    for (let c = 0; c < channels; c += 1) {
      ctx.strokeStyle = COLORS[c % COLORS.length];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let t = 0; t < timesteps; t += 1) {
        const idx = t * stride + c;
        const v = data[idx] ?? 0;
        const x = (t / Math.max(timesteps - 1, 1)) * width;
        const y = height - ((v - min) / range) * height;
        if (t === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    if (labels && labels.length) {
      ctx.font = "12px 'Avenir Next', sans-serif";
      labels.slice(0, channels).forEach((label, idx) => {
        ctx.fillStyle = COLORS[idx % COLORS.length];
        ctx.fillText(label, 8, 14 + idx * 14);
      });
    }
  }, [data, shape, width, height, labels]);

  return <canvas ref={canvasRef} className="nnvis-lineplot" />;
}
