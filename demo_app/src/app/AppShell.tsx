import React from "react";
import HeaderBar from "../components/layout/HeaderBar";
import LeftRail from "../components/layout/LeftRail";
import CenterStage from "../components/layout/CenterStage";
import RightInspector from "../components/layout/RightInspector";
import Waves from "../components/Waves";

export default function AppShell() {
  return (
    <div className="app-shell">
      <div className="bg-waves" aria-hidden>
        <div className="bg-waves-inner">
          <Waves
            lineColor="#2a7cff"
            backgroundColor="transparent"
            waveSpeedX={0.065}
            waveSpeedY={0.04}
            waveAmpX={22}
            waveAmpY={20}
            friction={0.6}
            tension={0.03}
            maxCursorMove={110}
            xGap={16}
            yGap={36}
          />
        </div>
      </div>

      <HeaderBar />
      <div className="dashboard-grid">
        <LeftRail />
        <CenterStage />
        <RightInspector />
      </div>
    </div>
  );
}
