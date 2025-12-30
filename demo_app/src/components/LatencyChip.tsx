import React from "react";

type Props = {
  value: number;
};

export function LatencyChip({ value }: Props) {
  return (
    <div className="chip">Latency {value.toFixed(1)} ms</div>
  );
}
