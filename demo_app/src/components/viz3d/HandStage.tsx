import React, { Suspense, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { shallow } from "zustand/shallow";
import { useDemoStore } from "../../state/useDemoStore";
import HandModel from "./HandModel";
import HandOverlayHUD from "./HandOverlayHUD";

type AdaptiveDprProps = {
  enabled: boolean;
  onLowPerfChange: (value: boolean) => void;
};

function AdaptiveDpr({ enabled, onLowPerfChange }: AdaptiveDprProps) {
  const { setDpr } = useThree();
  const lowPerfRef = useRef(false);
  const spikeCountRef = useRef(0);

  useFrame((_, delta) => {
    if (!enabled || lowPerfRef.current) return;
    const dt = Math.min(delta, 0.5);
    if (dt > 1 / 30) {
      spikeCountRef.current += 1;
    } else if (spikeCountRef.current > 0) {
      spikeCountRef.current -= 1;
    }

    if (spikeCountRef.current >= 6) {
      lowPerfRef.current = true;
      setDpr(1);
      onLowPerfChange(true);
    }
  });

  return null;
}

export default function HandStage() {
  const { tick, wsState, lastWsMessageAt } = useDemoStore(
    (state) => ({
      tick: state.tick,
      wsState: state.wsState,
      lastWsMessageAt: state.lastWsMessageAt
    }),
    shallow
  );

  // ====== SAFE MODE (GPU/memory defensive defaults) ======
  const SAFE_MODE = import.meta.env.VITE_SAFE_MODE === "1";
  const [lowPerf, setLowPerf] = useState(false);
  const qualityReduced = SAFE_MODE || lowPerf;

  const action = tick?.prediction.action_name ?? "REST";
  const finger = tick?.prediction.finger_name ?? "NONE";
  const actionConfidence = tick?.prediction.action_confidence ?? 0;
  const fingerConfidence = tick?.prediction.finger_confidence ?? 0;
  const confidence = Math.max(actionConfidence, fingerConfidence);
  const latencyMs = tick?.diagnostics.latency_ms ?? 0;

  const glProps = useMemo(
    () => ({
      antialias: !qualityReduced,
      powerPreference: "low-power" as const,
      preserveDrawingBuffer: false
    }),
    [qualityReduced]
  );

  return (
    <div className="hand-stage">
      <div className="hand-canvas">
        <Canvas
          dpr={qualityReduced ? 1 : [1, 1.5]}
          gl={glProps}
          camera={{ position: [0, 1.4, 3.4], fov: 46 }}
        >
          <color attach="background" args={["#0a121b"]} />

          <AdaptiveDpr enabled={!qualityReduced} onLowPerfChange={setLowPerf} />

          {/* Lighting: keep it simple in safe/low-perf mode */}
          <ambientLight intensity={qualityReduced ? 0.65 : 0.55} />
          <directionalLight position={[4, 6, 3]} intensity={qualityReduced ? 0.9 : 1.1} />
          {!qualityReduced && <directionalLight position={[-6, 4, -6]} intensity={0.65} />}
          {!qualityReduced && <pointLight position={[-4, -2, -4]} intensity={0.6} />}

          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.62, 0]}>
            <planeGeometry args={[12, 12]} />
            <meshStandardMaterial
              color="#0b1520"
              roughness={0.95}
              metalness={0}
              opacity={0.55}
              transparent
            />
          </mesh>

          <Suspense fallback={null}>
            <HandModel
              action={action}
              finger={finger}
              actionConfidence={actionConfidence}
              fingerConfidence={fingerConfidence}
              confidence={confidence}
              safeMode={SAFE_MODE}
              lowPerfMode={lowPerf}
            />
          </Suspense>

          {/* Controls disabled in SAFE_MODE to reduce CPU/GPU churn */}
          {!qualityReduced && (
            <OrbitControls
              enablePan={false}
              minDistance={2.2}
              maxDistance={5.5}
              minPolarAngle={0.35}
              maxPolarAngle={1.6}
              rotateSpeed={0.6}
            />
          )}
        </Canvas>

        <HandOverlayHUD
          action={action}
          finger={finger}
          confidence={confidence}
          latencyMs={latencyMs}
          wsState={wsState}
          lastMessageAt={lastWsMessageAt}
        />
      </div>
    </div>
  );
}
