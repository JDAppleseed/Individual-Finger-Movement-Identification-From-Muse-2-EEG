import React, { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import {
  Bone,
  Box3,
  Color,
  Group,
  MathUtils,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  SkinnedMesh,
  SphereGeometry,
  Vector3
} from "three";

type Props = {
  action: string;
  finger: string;
  confidence: number;
  actionConfidence: number;
  fingerConfidence: number;
  safeMode?: boolean;
  lowPerfMode?: boolean;
};

// Keep the stage usable if GLB loading fails at runtime.
type ErrorBoundaryProps = { fallback: React.ReactNode; children: React.ReactNode; resetKey?: string };
type ErrorBoundaryState = { hasError: boolean };

class GltfFallbackBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps) {
    if (prevProps.resetKey !== this.props.resetKey && this.state.hasError) {
      this.setState({ hasError: false });
    }
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

type FingerKey = "THUMB" | "INDEX" | "MIDDLE" | "RING" | "PINKY";
type FingerName = FingerKey | "NONE";
type ActionName = "OPEN" | "CLOSE" | "REST";

type CurlAxis = "x" | "y" | "z";

const ACCENT = new Color("#41f2c2");
const BASE = new Color("#7f94ae");
const DIM = new Color("#2b3442");
const NO_EMISSIVE = new Color(0x000000);

const FINGER_ORDER: FingerKey[] = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"];
const FINGER_TOKENS: Record<FingerKey, string[]> = {
  THUMB: ["thumb"],
  INDEX: ["index"],
  MIDDLE: ["middle"],
  RING: ["ring"],
  PINKY: ["pinky", "pinkie"]
};

const FINGER_BIAS: Record<FingerKey, number> = {
  THUMB: 0.92,
  INDEX: 1.0,
  MIDDLE: 1.02,
  RING: 0.97,
  PINKY: 0.9
};

const ACTION_CONF_THRESHOLD = 0.55;
const FINGER_CONF_THRESHOLD = 0.6;
const FINGER_HOLD_SECONDS = 0.18;
const FINGER_RELEASE_SECONDS = 0.08;
const IDLE_BREATH_FREQ = 1.4;
const IDLE_BREATH_AMPLITUDE = 0.025;

const DEFAULT_CURL_AXIS: CurlAxis = "z";
const FINGER_CURL_AXIS_OVERRIDE: Partial<Record<FingerKey, CurlAxis>> = {};

const DEBUG_HAND = import.meta.env.VITE_HAND_DEBUG === "1";
const TEST_ANIM = import.meta.env.VITE_HAND_TEST_ANIM === "1";

// You will likely replace these names after you print the bone list.
// Keeping them here is fine for now.
const RIG_BONE_CHAINS_BY_NAME: Record<FingerKey, string[]> = {
  THUMB: ["Bone001", "Bone002", "Bone003", "Bone004"],
  INDEX: ["Bone005", "Bone006", "Bone007", "Bone008"],
  MIDDLE: ["Bone009", "Bone010", "Bone011", "Bone012"],
  RING: ["Bone013", "Bone014", "Bone015", "Bone016"],
  PINKY: ["Bone017", "Bone018", "Bone019"]
};

function normalizeAction(value: string): ActionName {
  const v = (value ?? "").toUpperCase();
  if (v.includes("OPEN")) return "OPEN";
  if (v.includes("CLOSE")) return "CLOSE";
  if (v.includes("REST")) return "REST";
  return "REST";
}

function normalizeFinger(value: string): FingerName {
  const v = (value ?? "").toUpperCase();
  if (v.includes("THUMB")) return "THUMB";
  if (v.includes("INDEX")) return "INDEX";
  if (v.includes("MIDDLE")) return "MIDDLE";
  if (v.includes("RING")) return "RING";
  if (v.includes("PINKY") || v.includes("PINKIE")) return "PINKY";
  return "NONE";
}

function fingerKeyFromName(name: string): FingerKey | null {
  const lower = name.toLowerCase();
  for (const key of FINGER_ORDER) {
    if (FINGER_TOKENS[key].some((token) => lower.includes(token))) {
      return key;
    }
  }
  return null;
}

function normalizeBoneName(name: string) {
  return name.replace(/\./g, "").toLowerCase();
}

function findBoneByName(skeletonBones: Bone[], name: string): Bone | null {
  const target = normalizeBoneName(name);
  return (
    skeletonBones.find((bone) => normalizeBoneName(bone.name) === target) ?? null
  );
}

function getCurlAxis(finger: FingerKey): CurlAxis {
  return FINGER_CURL_AXIS_OVERRIDE[finger] ?? DEFAULT_CURL_AXIS;
}

// Favor distal curl over proximal for a more natural flex.
function segmentWeight(count: number, index: number) {
  if (count <= 1) return 0.7;
  const t = index / (count - 1);
  return MathUtils.lerp(0.35, 0.9, t);
}

// Map action -> relaxed/open/closed curl targets.
function targetFlexFromAction(action: ActionName) {
  if (action === "CLOSE") return 1.0;
  if (action === "OPEN") return 0.05;
  return 0.28; // REST = slightly contracted
}

function useModelAvailable(url: string) {
  const [available, setAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    fetch(url, { method: "HEAD" })
      .then((res) => {
        if (!active) return;
        setAvailable(res.ok);
      })
      .catch(() => {
        if (!active) return;
        setAvailable(false);
      });

    return () => {
      active = false;
    };
  }, [url]);

  return available;
}

type FingerWeights = Record<FingerKey, number>;

function computeFingerInfluence(mesh: SkinnedMesh, fingerBoneIndices: Record<FingerKey, Set<number>>) {
  const skinIndex = mesh.geometry.getAttribute("skinIndex");
  const skinWeight = mesh.geometry.getAttribute("skinWeight");
  if (!skinIndex || !skinWeight) return null;

  const fingerWeights: FingerWeights = { THUMB: 0, INDEX: 0, MIDDLE: 0, RING: 0, PINKY: 0 };
  let totalWeight = 0;

  const indexToFingers = new Map<number, FingerKey[]>();
  FINGER_ORDER.forEach((fingerName) => {
    fingerBoneIndices[fingerName].forEach((index) => {
      const existing = indexToFingers.get(index);
      if (existing) existing.push(fingerName);
      else indexToFingers.set(index, [fingerName]);
    });
  });

  for (let i = 0; i < skinIndex.count; i += 1) {
    const indices = [skinIndex.getX(i), skinIndex.getY(i), skinIndex.getZ(i), skinIndex.getW(i)];
    const weights = [skinWeight.getX(i), skinWeight.getY(i), skinWeight.getZ(i), skinWeight.getW(i)];

    for (let j = 0; j < 4; j += 1) {
      const weight = weights[j];
      if (weight <= 0) continue;
      totalWeight += weight;
      const fingers = indexToFingers.get(indices[j]);
      if (!fingers) continue;
      fingers.forEach((fingerName) => {
        fingerWeights[fingerName] += weight;
      });
    }
  }

  return { fingerWeights, totalWeight };
}

type MotionState = {
  baseFlex: number;
  fingerFlex: Record<FingerKey, number>;
  stableFinger: FingerName;
  candidateFinger: FingerName;
  candidateHold: number;
  fingerWeight: number;
  intensity: number;
};

function initMotionState(): MotionState {
  return {
    baseFlex: targetFlexFromAction("REST"),
    fingerFlex: {
      THUMB: targetFlexFromAction("REST"),
      INDEX: targetFlexFromAction("REST"),
      MIDDLE: targetFlexFromAction("REST"),
      RING: targetFlexFromAction("REST"),
      PINKY: targetFlexFromAction("REST")
    },
    stableFinger: "NONE",
    candidateFinger: "NONE",
    candidateHold: 0,
    fingerWeight: 0,
    intensity: 0
  };
}

function updateMotionState(
  state: MotionState,
  action: string,
  finger: string,
  actionConfidence: number,
  fingerConfidence: number,
  dt: number,
  elapsed: number
): MotionState {
  let actionName = normalizeAction(action);
  let actionWeight = MathUtils.clamp(
    (actionConfidence - ACTION_CONF_THRESHOLD) / (1 - ACTION_CONF_THRESHOLD),
    0,
    1
  );
  let fingerName = normalizeFinger(finger);
  let fingerWeightRaw = MathUtils.clamp(
    (fingerConfidence - FINGER_CONF_THRESHOLD) / (1 - FINGER_CONF_THRESHOLD),
    0,
    1
  );

  if (TEST_ANIM) {
    actionName = Math.sin(elapsed * 1.2) >= 0 ? "CLOSE" : "OPEN";
    actionWeight = 1;
    const fingerIndex = Math.floor(elapsed * 0.6) % FINGER_ORDER.length;
    fingerName = FINGER_ORDER[fingerIndex];
    fingerWeightRaw = 1;
    state.stableFinger = fingerName;
    state.candidateFinger = fingerName;
    state.candidateHold = FINGER_HOLD_SECONDS;
  } else {
    const candidate = fingerWeightRaw > 0 ? fingerName : "NONE";
    if (candidate !== state.candidateFinger) {
      state.candidateFinger = candidate;
      state.candidateHold = 0;
    } else {
      state.candidateHold += dt;
    }

    const holdTime = candidate === "NONE" ? FINGER_RELEASE_SECONDS : FINGER_HOLD_SECONDS;
    if (state.candidateHold >= holdTime) {
      state.stableFinger = candidate;
    }
  }

  const actionTarget = MathUtils.lerp(targetFlexFromAction("REST"), targetFlexFromAction(actionName), actionWeight);
  const idle = IDLE_BREATH_AMPLITUDE * (0.5 + 0.5 * Math.sin(elapsed * IDLE_BREATH_FREQ));
  state.baseFlex = MathUtils.damp(state.baseFlex, actionTarget + idle, 5.5, dt);

  state.fingerWeight = MathUtils.damp(state.fingerWeight, fingerWeightRaw, 6, dt);
  const stableFinger = state.stableFinger;
  const fingerWeight = stableFinger === "NONE" ? 0 : state.fingerWeight;

  const baseFlex = MathUtils.clamp(state.baseFlex, 0.02, 1.35);
  const sympathetic = baseFlex * (0.04 + 0.08 * actionWeight);

  FINGER_ORDER.forEach((key) => {
    const bias = FINGER_BIAS[key];
    let targetFlex = baseFlex * bias;

    if (stableFinger === key) {
      if (actionName === "OPEN") {
        targetFlex = baseFlex * (1 - fingerWeight * 0.7);
      } else if (actionName === "CLOSE") {
        targetFlex = baseFlex * (1 + fingerWeight * 0.6);
      } else {
        targetFlex = baseFlex * (1 + fingerWeight * 0.25);
      }
    } else {
      targetFlex = baseFlex * bias + sympathetic;
    }

    targetFlex = MathUtils.clamp(targetFlex, 0.02, 1.45);
    state.fingerFlex[key] = MathUtils.damp(state.fingerFlex[key], targetFlex, 7.5, dt);
  });

  state.intensity = TEST_ANIM ? 1 : MathUtils.clamp(Math.max(actionConfidence, fingerConfidence), 0, 1);

  return state;
}

function ProceduralHand({ action, finger, actionConfidence, fingerConfidence }: Props) {
  const motionRef = useRef<MotionState>(initMotionState());
  const segmentRefs = useRef<Record<FingerKey, Mesh[]>>({
    THUMB: [],
    INDEX: [],
    MIDDLE: [],
    RING: [],
    PINKY: []
  });

  const materials = useMemo<Record<FingerKey | "palm", MeshStandardMaterial>>(() => {
    const base = new MeshStandardMaterial({ color: BASE.clone(), roughness: 0.45, metalness: 0.1 });
    return {
      palm: base,
      THUMB: base.clone(),
      INDEX: base.clone(),
      MIDDLE: base.clone(),
      RING: base.clone(),
      PINKY: base.clone()
    };
  }, []);

  const fingerDefs = useMemo(
    () => [
      { name: "THUMB" as FingerKey, position: new Vector3(-0.75, 0.2, 0.25), rotation: [0, 0, 0.6] },
      { name: "INDEX" as FingerKey, position: new Vector3(-0.35, 0.35, 0.45), rotation: [0, 0, 0.15] },
      { name: "MIDDLE" as FingerKey, position: new Vector3(0, 0.4, 0.5), rotation: [0, 0, 0] },
      { name: "RING" as FingerKey, position: new Vector3(0.35, 0.35, 0.45), rotation: [0, 0, -0.08] },
      { name: "PINKY" as FingerKey, position: new Vector3(0.7, 0.25, 0.35), rotation: [0, 0, -0.18] }
    ],
    []
  );

  useFrame((state, delta) => {
    const dt = Math.min(delta, 1 / 20);
    const motion = updateMotionState(
      motionRef.current,
      action,
      finger,
      actionConfidence,
      fingerConfidence,
      dt,
      state.clock.elapsedTime
    );

    FINGER_ORDER.forEach((name) => {
      const segments = segmentRefs.current[name] ?? [];
      const fingerFlex = motion.fingerFlex[name];

      segments.forEach((segment, sIdx) => {
        if (!segment) return;
        segment.rotation.x = -fingerFlex * segmentWeight(segments.length, sIdx);
      });

      const mat = materials[name];
      if (mat) {
        const isActive = motion.stableFinger === name;
        const mix = isActive ? 0.35 + motion.intensity * 0.55 : 0.12;
        mat.color.copy(DIM).lerp(isActive ? ACCENT : BASE, mix);
        mat.emissive.copy(ACCENT).multiplyScalar(isActive ? 0.3 + motion.intensity * 0.6 : 0.05);
      }
    });
  });

  return (
    <group position={[0, -0.3, 0]} rotation={[0.05, 0, 0]}>
      <mesh material={materials.palm} position={[0, 0, 0]}>
        <boxGeometry args={[2.0, 0.5, 1.2]} />
      </mesh>

      {fingerDefs.map((fingerDef) => (
        <group
          key={fingerDef.name}
          position={fingerDef.position.toArray()}
          rotation={fingerDef.rotation as [number, number, number]}
        >
          {[0, 1, 2].map((idx) => (
            <mesh
              key={`${fingerDef.name}-seg-${idx}`}
              ref={(node) => {
                if (!node) return;
                segmentRefs.current[fingerDef.name][idx] = node;
              }}
              material={materials[fingerDef.name]}
              position={[0, 0.25 + idx * 0.3, 0]}
            >
              <boxGeometry args={[0.28, 0.32, 0.28]} />
            </mesh>
          ))}
        </group>
      ))}
    </group>
  );
}

function GLTFHand({
  action,
  finger,
  actionConfidence,
  fingerConfidence,
  safeMode,
  lowPerfMode,
  url,
  onRigStatus
}: Props & { url: string; onRigStatus?: (ok: boolean) => void }) {
  const gltf = useGLTF(url) as any;

  const motionRef = useRef<MotionState>(initMotionState());
  const groupRef = useRef<Group | null>(null);
  const tipRefs = useRef<Record<FingerKey, Mesh | null>>({
    THUMB: null,
    INDEX: null,
    MIDDLE: null,
    RING: null,
    PINKY: null
  });
  const scratch = useMemo(() => new Vector3(), []);
  const bindPoseRef = useRef<WeakMap<Bone, { x: number; y: number; z: number }>>(new WeakMap());
  const mappingLoggedRef = useRef(false);
  const missingLoggedRef = useRef(false);

  // Track cloned materials so we can dispose on unmount.
  const clonedMaterialsRef = useRef<MeshStandardMaterial[]>([]);

  const rig = useMemo(() => {
    const allMeshes: Mesh[] = [];
    const skinnedMeshes: SkinnedMesh[] = [];
    const fingerGroups: Record<FingerKey, Object3D[]> = {
      THUMB: [],
      INDEX: [],
      MIDDLE: [],
      RING: [],
      PINKY: []
    };

    gltf.scene.traverse((obj: Object3D) => {
      if ((obj as Mesh).isMesh) allMeshes.push(obj as Mesh);
      if ((obj as SkinnedMesh).isSkinnedMesh) skinnedMeshes.push(obj as SkinnedMesh);
      const key = fingerKeyFromName(obj.name);
      if (key) fingerGroups[key].push(obj);
    });

    const primarySkeleton = skinnedMeshes.find((mesh) => !!mesh.skeleton)?.skeleton ?? null;
    const skeletonBones = primarySkeleton?.bones ?? [];
    const skeletonBoneNames = skeletonBones.map((bone) => bone.name);

    const boneSet = new Set<Bone>();
    skinnedMeshes.forEach((mesh) => mesh.skeleton?.bones.forEach((bone) => boneSet.add(bone)));

    const explicitChains: Record<FingerKey, Bone[]> = {
      THUMB: [],
      INDEX: [],
      MIDDLE: [],
      RING: [],
      PINKY: []
    };

    const missingBones: Record<FingerKey, string[]> = {
      THUMB: [],
      INDEX: [],
      MIDDLE: [],
      RING: [],
      PINKY: []
    };

    if (skeletonBones.length) {
      FINGER_ORDER.forEach((fingerKey) => {
        const names = RIG_BONE_CHAINS_BY_NAME[fingerKey];
        explicitChains[fingerKey] = names
          .map((name) => {
            const bone = findBoneByName(skeletonBones, name);
            if (!bone) missingBones[fingerKey].push(name);
            return bone;
          })
          .filter((bone): bone is Bone => !!bone);
      });
    } else {
      FINGER_ORDER.forEach((fingerKey) => {
        missingBones[fingerKey].push(...RIG_BONE_CHAINS_BY_NAME[fingerKey]);
      });
    }

    const fingerChains = explicitChains;

    const fingerAxes: Record<FingerKey, CurlAxis> = {
      THUMB: getCurlAxis("THUMB"),
      INDEX: getCurlAxis("INDEX"),
      MIDDLE: getCurlAxis("MIDDLE"),
      RING: getCurlAxis("RING"),
      PINKY: getCurlAxis("PINKY")
    };

    const fingerTips: Record<FingerKey, Bone | null> = {
      THUMB: fingerChains.THUMB.length ? fingerChains.THUMB[fingerChains.THUMB.length - 1] : null,
      INDEX: fingerChains.INDEX.length ? fingerChains.INDEX[fingerChains.INDEX.length - 1] : null,
      MIDDLE: fingerChains.MIDDLE.length ? fingerChains.MIDDLE[fingerChains.MIDDLE.length - 1] : null,
      RING: fingerChains.RING.length ? fingerChains.RING[fingerChains.RING.length - 1] : null,
      PINKY: fingerChains.PINKY.length ? fingerChains.PINKY[fingerChains.PINKY.length - 1] : null
    };

    const fingerMeshes: Record<FingerKey, Set<Mesh>> = {
      THUMB: new Set(),
      INDEX: new Set(),
      MIDDLE: new Set(),
      RING: new Set(),
      PINKY: new Set()
    };

    // Mesh highlight discovery (safe, but can be expensive—kept as-is)
    FINGER_ORDER.forEach((name) => {
      fingerGroups[name].forEach((obj) => {
        obj.traverse((child) => {
          if ((child as Mesh).isMesh) fingerMeshes[name].add(child as Mesh);
        });
      });
    });

    skinnedMeshes.forEach((mesh) => {
      const skeleton = mesh.skeleton;
      if (!skeleton) return;

      const boneIndices: Record<FingerKey, Set<number>> = {
        THUMB: new Set(),
        INDEX: new Set(),
        MIDDLE: new Set(),
        RING: new Set(),
        PINKY: new Set()
      };

      skeleton.bones.forEach((bone, index) => {
        const key = fingerKeyFromName(bone.name);
        if (key) boneIndices[key].add(index);
      });

      const influence = computeFingerInfluence(mesh, boneIndices);
      if (!influence || influence.totalWeight <= 0) return;

      FINGER_ORDER.forEach((fingerName) => {
        const share = influence.fingerWeights[fingerName] / influence.totalWeight;
        if (share >= 0.35) fingerMeshes[fingerName].add(mesh);
      });
    });

    const hasBoneChains = FINGER_ORDER.every(
      (fingerKey) => fingerChains[fingerKey].length === RIG_BONE_CHAINS_BY_NAME[fingerKey].length
    );
    const hasMissingBones = FINGER_ORDER.some((fingerKey) => missingBones[fingerKey].length > 0);

    return {
      allMeshes,
      fingerGroups,
      fingerChains,
      fingerAxes,
      fingerTips,
      fingerMeshes,
      hasRig: skeletonBones.length > 0 && boneSet.size > 0,
      hasBoneChains,
      missingBones,
      skeletonBoneNames,
      hasMissingBones
    };
  }, [gltf]);

  const layout = useMemo(() => {
    const box = new Box3().setFromObject(gltf.scene);
    const size = new Vector3();
    const center = new Vector3();
    box.getSize(size);
    box.getCenter(center);
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const scale = MathUtils.clamp(1.2 / maxDim, 0.7, 1.6) * 0.5;
    const offset = new Vector3(-center.x, -box.min.y, -center.z);
    return { scale, offset };
  }, [gltf]);

  // ====== DEFENSE: clone materials exactly once per mesh, and dispose on unmount ======
  useEffect(() => {
    const cloned: MeshStandardMaterial[] = [];

    rig.allMeshes.forEach((mesh) => {
      const ud = (mesh as any).userData ?? ((mesh as any).userData = {});
      if (ud._matCloned) return;
      ud._matCloned = true;

      if (Array.isArray(mesh.material)) {
        const next = mesh.material.map((mat) => {
          const c = (mat as MeshStandardMaterial).clone();
          cloned.push(c);
          return c;
        }) as MeshStandardMaterial[];
        mesh.material = next;
      } else if (mesh.material) {
        const c = (mesh.material as MeshStandardMaterial).clone();
        cloned.push(c);
        mesh.material = c;
      }
    });

    // Replace list each mount (dispose previous if any)
    clonedMaterialsRef.current.forEach((m) => {
      try {
        m.dispose();
      } catch {}
    });
    clonedMaterialsRef.current = cloned;

    return () => {
      clonedMaterialsRef.current.forEach((m) => {
        try {
          m.dispose();
        } catch {}
      });
      clonedMaterialsRef.current = [];
    };
  }, [rig.allMeshes]);

  useEffect(() => {
    if (!rig.hasRig || !rig.hasBoneChains) return;
    bindPoseRef.current = new WeakMap();
    FINGER_ORDER.forEach((name) => {
      rig.fingerChains[name].forEach((bone) => {
        bindPoseRef.current.set(bone, {
          x: bone.rotation.x,
          y: bone.rotation.y,
          z: bone.rotation.z
        });
      });
    });
  }, [rig]);

  useEffect(() => {
    if (!DEBUG_HAND || mappingLoggedRef.current) return;
    console.info("[HandRig] bones\n" + rig.skeletonBoneNames.join("\n"));
    const mapping: Record<FingerKey, string[]> = {
      THUMB: rig.fingerChains.THUMB.map((bone) => bone.name),
      INDEX: rig.fingerChains.INDEX.map((bone) => bone.name),
      MIDDLE: rig.fingerChains.MIDDLE.map((bone) => bone.name),
      RING: rig.fingerChains.RING.map((bone) => bone.name),
      PINKY: rig.fingerChains.PINKY.map((bone) => bone.name)
    };
    console.info("[HandRig] mapping", { mapping, axes: rig.fingerAxes });
    mappingLoggedRef.current = true;
  }, [rig]);

  useEffect(() => {
    if (rig.hasBoneChains || missingLoggedRef.current) return;
    const missingSummary = FINGER_ORDER.map((fingerKey) => {
      const missing = rig.missingBones[fingerKey];
      if (!missing.length) return null;
      return `${fingerKey}: ${missing.join(", ")}`;
    })
      .filter((entry) => entry)
      .join(" | ");
    const details = missingSummary || "missing bone names";
    console.error(
      `[HandRig] Missing required bones (${details}). Falling back to ProceduralHand.`
    );
    missingLoggedRef.current = true;
  }, [rig]);

  useEffect(() => {
    if (!DEBUG_HAND || safeMode || lowPerfMode || !rig.hasRig) return;
    const geometry = new SphereGeometry(0.012, 10, 10);
    const material = new MeshStandardMaterial({ color: "#ffb347", emissive: "#ffb347" });
    const markers: Mesh[] = [];

    FINGER_ORDER.forEach((fingerKey) => {
      rig.fingerChains[fingerKey].forEach((bone) => {
        const marker = new Mesh(geometry, material);
        marker.userData._debugMarker = true;
        bone.add(marker);
        markers.push(marker);
      });
    });

    return () => {
      markers.forEach((marker) => {
        marker.parent?.remove(marker);
      });
      geometry.dispose();
      material.dispose();
    };
  }, [rig, safeMode, lowPerfMode]);

  useEffect(() => {
    if (!onRigStatus) return;
    onRigStatus(rig.hasRig && rig.hasBoneChains);
  }, [onRigStatus, rig]);

  // ====== Animation loop (defensive delta clamp; NO root tilt fallbacks) ======
  useFrame((state, delta) => {
    const dt = Math.min(delta, 1 / 20);
    const motion = updateMotionState(
      motionRef.current,
      action,
      finger,
      actionConfidence,
      fingerConfidence,
      dt,
      state.clock.elapsedTime
    );

    if (rig.hasBoneChains) {
      FINGER_ORDER.forEach((name) => {
        const chain = rig.fingerChains[name];
        if (!chain.length) return;
        const axis = rig.fingerAxes[name];
        const fingerFlex = motion.fingerFlex[name];

        chain.forEach((bone, boneIndex) => {
          const bind = bindPoseRef.current.get(bone);
          if (!bind) return;

          const amount = fingerFlex * segmentWeight(chain.length, boneIndex);
          bone.rotation.x = bind.x;
          bone.rotation.y = bind.y;
          bone.rotation.z = bind.z;
          if (axis === "x") bone.rotation.x = bind.x + amount;
          if (axis === "y") bone.rotation.y = bind.y + amount;
          if (axis === "z") bone.rotation.z = bind.z + amount;
        });
      });
    }

    // Highlighting (emissive only; safe)
    const highlightMeshes = motion.stableFinger === "NONE" ? null : rig.fingerMeshes[motion.stableFinger];
    const hasMeshHighlight = !!highlightMeshes && highlightMeshes.size > 0;

    rig.allMeshes.forEach((mesh) => {
      const isHighlighted = hasMeshHighlight && highlightMeshes?.has(mesh);
      const targetColor = isHighlighted ? ACCENT : NO_EMISSIVE;
      const targetIntensity = isHighlighted ? 0.3 + motion.intensity * 0.7 : 0;

      if (Array.isArray(mesh.material)) {
        mesh.material.forEach((mat) => {
          const material = mat as MeshStandardMaterial;
          material.emissive.copy(targetColor);
          material.emissiveIntensity = targetIntensity;
        });
      } else {
        const material = mesh.material as MeshStandardMaterial;
        material.emissive.copy(targetColor);
        material.emissiveIntensity = targetIntensity;
      }
    });

    // Tip markers
    const group = groupRef.current;
    const tipsEnabled = !safeMode && !lowPerfMode;
    if (group) {
      FINGER_ORDER.forEach((name) => {
        const marker = tipRefs.current[name];
        if (!marker) return;
        if (!tipsEnabled) {
          marker.visible = false;
          return;
        }
        const tip = rig.fingerTips[name];
        if (!tip || motion.stableFinger !== name) {
          marker.visible = false;
          return;
        }
        tip.getWorldPosition(scratch);
        group.worldToLocal(scratch);
        marker.position.copy(scratch);
        marker.visible = true;
        const material = marker.material as MeshStandardMaterial;
        material.emissiveIntensity = 0.35 + motion.intensity * 0.75;
      });
    }
  });

  return (
    <group
      ref={groupRef}
      scale={layout.scale}
      position={[0, -0.62, 0]}
      rotation={[0, Math.PI / 2, 0]}   // ✅ 90° fix (try Y first)
    >
      <group position={[layout.offset.x, layout.offset.y, layout.offset.z]}>
        <primitive object={gltf.scene} />
      </group>

      {rig.hasRig &&
        FINGER_ORDER.map((name) => (
          <mesh
            key={`${name}-tip`}
            ref={(node) => {
              tipRefs.current[name] = node;
            }}
            visible={false}
          >
            <sphereGeometry args={[0.045, 18, 18]} />
            <meshStandardMaterial
              color="#41f2c2"
              emissive="#41f2c2"
              emissiveIntensity={0.7}
              transparent
              opacity={0.85}
            />
          </mesh>
        ))}
    </group>
  );
}

export default function HandModel({
  action,
  finger,
  confidence,
  actionConfidence,
  fingerConfidence,
  safeMode,
  lowPerfMode
}: Props) {
  const url = "/models/hand.glb";
  const available = useModelAvailable(url);
  const [rigInvalid, setRigInvalid] = useState(false);

  useEffect(() => {
    setRigInvalid(false);
  }, [url]);

  if (rigInvalid || available === false || available === null) {
    return (
      <ProceduralHand
        action={action}
        finger={finger}
        confidence={confidence}
        actionConfidence={actionConfidence}
        fingerConfidence={fingerConfidence}
      />
    );
  }

  return (
    <GltfFallbackBoundary
      fallback={
        <ProceduralHand
          action={action}
          finger={finger}
          confidence={confidence}
          actionConfidence={actionConfidence}
          fingerConfidence={fingerConfidence}
        />
      }
      resetKey={url}
    >
      <GLTFHand
        action={action}
        finger={finger}
        confidence={confidence}
        actionConfidence={actionConfidence}
        fingerConfidence={fingerConfidence}
        safeMode={safeMode}
        lowPerfMode={lowPerfMode}
        url={url}
        onRigStatus={(ok) => {
          if (!ok) setRigInvalid(true);
        }}
      />
    </GltfFallbackBoundary>
  );
}
