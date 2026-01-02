import React, { useEffect, useRef } from "react";
import { computeMaxAbs, valueToColor } from "./utils";

export type ConvKernelGridProps = {
  data: Float32Array;
  shape: [number, number, number];
  width?: number;
  height?: number;
};

const ConvKernelGrid = React.forwardRef<HTMLCanvasElement, ConvKernelGridProps>(({ data, shape, width = 480, height = 240 }, ref) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    const [outCh, inCh, k] = shape;
    const cols = Math.ceil(Math.sqrt(outCh));
    const rows = Math.ceil(outCh / cols);
    const cellW = width / cols;
    const cellH = height / rows;
    const maxAbs = computeMaxAbs(data);

    for (let o = 0; o < outCh; o += 1) {
      const gridX = o % cols;
      const gridY = Math.floor(o / cols);
      const originX = gridX * cellW;
      const originY = gridY * cellH;
      const kernelW = cellW / Math.max(k, 1);
      const kernelH = cellH / Math.max(inCh, 1);

      for (let i = 0; i < inCh; i += 1) {
        for (let t = 0; t < k; t += 1) {
          const idx = o * inCh * k + i * k + t;
          const value = data[idx] ?? 0;
          ctx.fillStyle = valueToColor(value, maxAbs);
          ctx.fillRect(originX + t * kernelW, originY + i * kernelH, kernelW, kernelH);
        }
      }

      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.strokeRect(originX + 0.5, originY + 0.5, cellW - 1, cellH - 1);
    }
  }, [data, shape, width, height]);

  return (
    <canvas
      ref={(node) => {
        canvasRef.current = node;
        if (typeof ref === "function") ref(node);
        else if (ref) ref.current = node;
      }}
      className="nnvis-heatmap"
    />
  );
});

ConvKernelGrid.displayName = "ConvKernelGrid";

export default ConvKernelGrid;
