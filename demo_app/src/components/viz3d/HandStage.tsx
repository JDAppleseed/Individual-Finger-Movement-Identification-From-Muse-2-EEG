import React, { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { shallow } from "zustand/shallow";
import { useDemoStore } from "../../state/useDemoStore";
import HandModel from "./HandModel";
import HandOverlayHUD from "./HandOverlayHUD";

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
  // Flip to false once the rig is stable.
  const SAFE_MODE = true;

  const action = tick?.prediction.action_name ?? "REST";
  const finger = tick?.prediction.finger_name ?? "NONE";
  const confidence = Math.max(
    tick?.prediction.action_confidence ?? 0,
    tick?.prediction.finger_confidence ?? 0
  );
  const latencyMs = tick?.diagnostics.latency_ms ?? 0;

  const glProps = useMemo(
    () => ({
      antialias: !SAFE_MODE,
      powerPreference: "low-power" as const,
      preserveDrawingBuffer: false
    }),
    [SAFE_MODE]
  );

  return (
    <div className="hand-stage">
      <div className="hand-canvas">
        <Canvas
          dpr={SAFE_MODE ? 1 : [1, 1.5]}
          gl={glProps}
          camera={{ position: [0, 1.4, 3.4], fov: 46 }}
        >
          <color attach="background" args={["#0a121b"]} />

          {/* Lighting: keep it simple in SAFE_MODE */}
          <ambientLight intensity={SAFE_MODE ? 0.65 : 0.55} />
          <directionalLight position={[4, 6, 3]} intensity={SAFE_MODE ? 0.9 : 1.1} />
          {!SAFE_MODE && <directionalLight position={[-6, 4, -6]} intensity={0.65} />}
          {!SAFE_MODE && <pointLight position={[-4, -2, -4]} intensity={0.6} />}

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
            <HandModel action={action} finger={finger} confidence={confidence} />
          </Suspense>

          {/* Controls disabled in SAFE_MODE to reduce CPU/GPU churn */}
          {!SAFE_MODE && (
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