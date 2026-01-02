import React, { useEffect, useRef } from "react";

export type BarChartProps = {
  data: number[];
  labels?: string[];
  width?: number;
  height?: number;
  highlightIndex?: number;
};

export default function BarChart({ data, labels, width = 280, height = 140, highlightIndex }: BarChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    const max = Math.max(...data, 1);
    const barW = width / Math.max(data.length, 1);
    data.forEach((value, idx) => {
      const h = (value / max) * (height - 20);
      ctx.fillStyle = idx === highlightIndex ? "#f97316" : "#38bdf8";
      ctx.fillRect(idx * barW + 4, height - h - 16, barW - 8, h);
      if (labels && labels[idx]) {
        ctx.fillStyle = "#cbd5f5";
        ctx.font = "10px 'Avenir Next', sans-serif";
        ctx.save();
        ctx.translate(idx * barW + barW / 2, height - 4);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(labels[idx], -barW / 4, 0);
        ctx.restore();
      }
    });
  }, [data, labels, width, height, highlightIndex]);

  return <canvas ref={canvasRef} className="nnvis-barchart" />;
}
