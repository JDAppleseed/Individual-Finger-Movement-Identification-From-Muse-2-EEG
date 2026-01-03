import React, { Suspense } from "react";
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

  const action = tick?.prediction.action_name ?? "REST";
  const finger = tick?.prediction.finger_name ?? "NONE";
  const confidence = Math.max(
    tick?.prediction.action_confidence ?? 0,
    tick?.prediction.finger_confidence ?? 0
  );
  const latencyMs = tick?.diagnostics.latency_ms ?? 0;

  return (
    <div className="hand-stage">
      <div className="hand-canvas">
        <Canvas
          dpr={[1, 1.5]}
          camera={{ position: [0, 1.4, 3.4], fov: 46 }}
        >
          <color attach="background" args={["#0a121b"]} />
          <ambientLight intensity={0.55} />
          <directionalLight position={[4, 6, 3]} intensity={1.1} />
          <directionalLight position={[-6, 4, -6]} intensity={0.65} />
          <pointLight position={[-4, -2, -4]} intensity={0.6} />
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.62, 0]}>
            <planeGeometry args={[12, 12]} />
            <meshStandardMaterial color="#0b1520" roughness={0.95} metalness={0} opacity={0.55} transparent />
          </mesh>
          <Suspense fallback={null}>
            <HandModel action={action} finger={finger} confidence={confidence} />
          </Suspense>
          <OrbitControls
            enablePan={false}
            minDistance={2.2}
            maxDistance={5.5}
            minPolarAngle={0.35}
            maxPolarAngle={1.6}
            rotateSpeed={0.6}
          />
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
