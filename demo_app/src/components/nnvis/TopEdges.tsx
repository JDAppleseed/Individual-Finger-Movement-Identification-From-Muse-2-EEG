import React, { useEffect, useRef } from "react";

export type TopEdge = { i: number; j: number; v: number };

export type TopEdgesProps = {
  edges: TopEdge[];
  shape: [number, number];
  width?: number;
  height?: number;
};

export default function TopEdges({ edges, shape, width = 260, height = 200 }: TopEdgesProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    const [rows, cols] = shape;
    const leftX = 20;
    const rightX = width - 20;

    ctx.strokeStyle = "rgba(148, 163, 184, 0.3)";
    ctx.beginPath();
    ctx.moveTo(leftX, 10);
    ctx.lineTo(leftX, height - 10);
    ctx.moveTo(rightX, 10);
    ctx.lineTo(rightX, height - 10);
    ctx.stroke();

    const maxAbs = Math.max(...edges.map((e) => Math.abs(e.v)), 1);
    edges.forEach((edge) => {
      const y1 = 10 + (edge.i / Math.max(rows - 1, 1)) * (height - 20);
      const y2 = 10 + (edge.j / Math.max(cols - 1, 1)) * (height - 20);
      const alpha = Math.min(1, Math.abs(edge.v) / maxAbs);
      ctx.strokeStyle = `rgba(248, 250, 252, ${0.1 + 0.9 * alpha})`;
      ctx.beginPath();
      ctx.moveTo(leftX, y1);
      ctx.lineTo(rightX, y2);
      ctx.stroke();
    });
  }, [edges, shape, width, height]);

  return <canvas ref={canvasRef} className="nnvis-topedges" />;
}
