import React from "react";

type Props = {
  value: number;
};

export function UncertaintyDial({ value }: Props) {
  const pct = Math.round(value * 100);
  return (
    <div className="card">
      <div className="card-title">Uncertainty</div>
      <div className="dial">
        <div className="dial-inner">{pct}%</div>
      </div>
    </div>
  );
}
