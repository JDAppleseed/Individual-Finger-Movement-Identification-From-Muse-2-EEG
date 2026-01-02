import React, { useEffect, useRef } from "react";
import { computeMaxAbs, valueToColor } from "./utils";

export type HeatmapHighlight = { row: number; col: number; color?: string };

export type HeatmapProps = {
  data: Float32Array;
  shape: [number, number];
  width?: number;
  height?: number;
  maxAbs?: number;
  highlights?: HeatmapHighlight[];
  showGrid?: boolean;
};

const Heatmap = React.forwardRef<HTMLCanvasElement, HeatmapProps>(
  ({ data, shape, width = 240, height = 160, maxAbs, highlights, showGrid = false }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      canvas.width = width;
      canvas.height = height;
      const [rows, cols] = shape;
      const cellW = width / Math.max(cols, 1);
      const cellH = height / Math.max(rows, 1);
      const scale = maxAbs ?? computeMaxAbs(data);
      ctx.clearRect(0, 0, width, height);
      for (let r = 0; r < rows; r += 1) {
        for (let c = 0; c < cols; c += 1) {
          const idx = r * cols + c;
          const value = data[idx] ?? 0;
          ctx.fillStyle = valueToColor(value, scale);
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
      }
      if (showGrid) {
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        for (let r = 0; r <= rows; r += 1) {
          ctx.beginPath();
          ctx.moveTo(0, r * cellH);
          ctx.lineTo(width, r * cellH);
          ctx.stroke();
        }
        for (let c = 0; c <= cols; c += 1) {
          ctx.beginPath();
          ctx.moveTo(c * cellW, 0);
          ctx.lineTo(c * cellW, height);
          ctx.stroke();
        }
      }
      if (highlights && highlights.length > 0) {
        for (const hl of highlights) {
          const x = hl.col * cellW;
          const y = hl.row * cellH;
          ctx.strokeStyle = hl.color ?? "rgba(255, 215, 0, 0.8)";
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x, y, cellW, cellH);
        }
      }
    }, [data, shape, width, height, maxAbs, highlights, showGrid]);

    return <canvas ref={(node) => {
      canvasRef.current = node;
      if (typeof ref === "function") ref(node);
      else if (ref) ref.current = node;
    }} className="nnvis-heatmap" />;
  }
);

Heatmap.displayName = "Heatmap";

export default Heatmap;
