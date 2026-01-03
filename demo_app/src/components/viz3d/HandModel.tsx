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
  Vector3
} from "three";

type Props = {
  action: string;
  finger: string;
  confidence: number;
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

// You will likely replace these names after you print the bone list.
// Keeping them here is fine for now.
const RIG_BONE_CHAINS_BY_NAME: Record<FingerKey, string[]> = {
  THUMB: ["Bone.001", "Bone.002", "Bone.003", "Bone.004"],
  INDEX: ["Bone.005", "Bone.006", "Bone.007", "Bone.008"],
  MIDDLE: ["Bone.009", "Bone.010", "Bone.011", "Bone.012"],
  RING: ["Bone.013", "Bone.014", "Bone.015", "Bone.016"],
  PINKY: ["Bone.017", "Bone.018", "Bone.019"]
};

function normalizeFinger(value: string): FingerName {
  const upper = value.toUpperCase();
  if (upper === "THUMB") return "THUMB";
  if (upper === "INDEX") return "INDEX";
  if (upper === "MIDDLE") return "MIDDLE";
  if (upper === "RING") return "RING";
  if (upper === "PINKY" || upper === "PINKIE") return "PINKY";
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

function findBoneByName(skeletonBones: Bone[], name: string): Bone | null {
  return skeletonBones.find((bone) => bone.name === name) ?? null;
}

// Favor distal curl over proximal for a more natural flex.
function segmentWeight(count: number, index: number) {
  if (count <= 1) return 0.7;
  const t = index / (count - 1);
  return MathUtils.lerp(0.35, 0.9, t);
}

// Map action -> relaxed/open/closed curl targets.
function targetFlexFromAction(action: string) {
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

function ProceduralHand({ action, finger, confidence }: Props) {
  const flexRef = useRef(0.35);
  const groupRef = useRef<Group | null>(null);
  const fingerRefs = useRef<Record<FingerKey, Group | null>>({
    THUMB: null,
    INDEX: null,
    MIDDLE: null,
    RING: null,
    PINKY: null
  });
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

  useFrame((_, delta) => {
    // Defensive clamp (avoid giant delta after tab sleeps / hiccups)
    const dt = Math.min(delta, 1 / 20);

    const target = targetFlexFromAction(action);
    flexRef.current = MathUtils.damp(flexRef.current, target, 6, dt);

    const highlighted = normalizeFinger(finger);
    const intensity = highlighted === "NONE" ? 0 : MathUtils.clamp(confidence, 0, 1);
    const isOpen = action === "OPEN";
    const isClose = action === "CLOSE";

    FINGER_ORDER.forEach((name, idx) => {
      const segments = segmentRefs.current[name] ?? [];
      const isActive = highlighted === name;
      const actionScale = isActive ? (isOpen ? 0.7 : isClose ? 1.3 : 1.15) : 1.0;
      const fingerFlex = flexRef.current * actionScale * (0.9 + idx * 0.04);

      segments.forEach((segment, sIdx) => {
        if (!segment) return;
        segment.rotation.x = -fingerFlex * segmentWeight(segments.length, sIdx);
      });

      const mat = materials[name];
      if (mat) {
        const mix = isActive ? 0.4 + intensity * 0.6 : 0.15;
        mat.color.copy(DIM).lerp(isActive ? ACCENT : BASE, mix);
        mat.emissive.copy(ACCENT).multiplyScalar(isActive ? 0.35 + intensity * 0.6 : 0.05);
      }
    });
  });

  return (
    <group ref={groupRef} position={[0, -0.3, 0]} rotation={[0.05, 0, 0]}>
      <mesh material={materials.palm} position={[0, 0, 0]}>
        <boxGeometry args={[2.0, 0.5, 1.2]} />
      </mesh>

      {fingerDefs.map((fingerDef) => (
        <group
          key={fingerDef.name}
          ref={(node) => {
            fingerRefs.current[fingerDef.name] = node;
          }}
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

function GLTFHand({ action, finger, confidence, url }: Props & { url: string }) {
  const gltf = useGLTF(url) as any;

  const flexRef = useRef(0.35);
  const groupRef = useRef<Group | null>(null);
  const tipRefs = useRef<Record<FingerKey, Mesh | null>>({
    THUMB: null,
    INDEX: null,
    MIDDLE: null,
    RING: null,
    PINKY: null
  });
  const scratch = useMemo(() => new Vector3(), []);

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

    const boneSet = new Set<Bone>();
    skinnedMeshes.forEach((mesh) => mesh.skeleton?.bones.forEach((bone) => boneSet.add(bone)));

    const explicitChains: Record<FingerKey, Bone[]> = {
      THUMB: [],
      INDEX: [],
      MIDDLE: [],
      RING: [],
      PINKY: []
    };

    if (skeletonBones.length) {
      FINGER_ORDER.forEach((fingerKey) => {
        explicitChains[fingerKey] = RIG_BONE_CHAINS_BY_NAME[fingerKey]
          .map((name) => findBoneByName(skeletonBones, name))
          .filter((bone): bone is Bone => !!bone);
      });
    }

    const fingerChains = explicitChains;

    const fingerAxes: Record<FingerKey, "x" | "z"> = {
      THUMB: "x",
      INDEX: "x",
      MIDDLE: "x",
      RING: "x",
      PINKY: "x"
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

    const hasBoneChains = FINGER_ORDER.every((fingerKey) => fingerChains[fingerKey].length > 0);

    return {
      allMeshes,
      fingerGroups,
      fingerChains,
      fingerAxes,
      fingerTips,
      fingerMeshes,
      hasRig: boneSet.size > 0,
      hasBoneChains
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

  // ====== OPTIONAL: DEV bone dump (SAFE: runs once on load) ======
  // Uncomment to verify bone names. Remove after you copy the list.
  /*
  useEffect(() => {
    const bones: string[] = [];
    gltf.scene.traverse((o: any) => {
      if (o?.isBone) bones.push(o.name);
    });
    console.log("HAND BONES:", bones);
  }, [gltf]);
  */

  // ====== Animation loop (defensive delta clamp; NO root tilt fallbacks) ======
  useFrame((_, delta) => {
    const dt = Math.min(delta, 1 / 20); // clamp large delta spikes
    const target = targetFlexFromAction(action);
    const intensity = MathUtils.clamp(confidence, 0, 1);
    const speed = 6 + intensity * 2;

    flexRef.current = MathUtils.damp(flexRef.current, target, speed, dt);
    const flex = flexRef.current * (1 + intensity * 0.08);

    const highlighted = normalizeFinger(finger);

    if (rig.hasBoneChains) {
      FINGER_ORDER.forEach((name, idx) => {
        const chain = rig.fingerChains[name];
        if (!chain.length) return;

        // Rest: slightly contracted; OPEN: more extension; CLOSE: more curl
        // Note: if OPEN/CLOSE come from action only (not finger), we still animate all fingers subtly,
        // but add extra emphasis on highlighted finger.
        const isActive = highlighted === name;
        const fingerBias = [0.92, 1.0, 1.02, 0.97, 0.9][idx] ?? 1.0;

        const emphasis = isActive ? (0.12 + 0.28 * intensity) : 0.0;
        const fingerFlex = flex * fingerBias + emphasis;

        chain.forEach((bone, boneIndex) => {
          const baseRot = bone.userData.baseRotation as { x: number; y: number; z: number } | undefined;
          if (!baseRot) bone.userData.baseRotation = { x: bone.rotation.x, y: bone.rotation.y, z: bone.rotation.z };
          const saved = bone.userData.baseRotation as { x: number; y: number; z: number };

          const amount = fingerFlex * segmentWeight(chain.length, boneIndex);

          // NOTE: Curl axis may not be X for your rig—this is addressed in brainstorm below.
          bone.rotation.x = saved.x - amount;
          bone.rotation.y = saved.y;
          bone.rotation.z = saved.z;
        });
      });
    } else {
      // DEFENSE: do nothing (do NOT rotate root/palm). Keep bind pose.
    }

    // Highlighting (emissive only; safe)
    const highlightMeshes = highlighted === "NONE" ? null : rig.fingerMeshes[highlighted];
    const hasMeshHighlight = !!highlightMeshes && highlightMeshes.size > 0;

    rig.allMeshes.forEach((mesh) => {
      const isHighlighted = hasMeshHighlight && highlightMeshes?.has(mesh);
      const targetColor = isHighlighted ? ACCENT : NO_EMISSIVE;
      const targetIntensity = isHighlighted ? 0.3 + intensity * 0.7 : 0;

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
    if (group) {
      FINGER_ORDER.forEach((name) => {
        const marker = tipRefs.current[name];
        if (!marker) return;
        const tip = rig.fingerTips[name];
        if (!tip || highlighted !== name) {
          marker.visible = false;
          return;
        }
        tip.getWorldPosition(scratch);
        group.worldToLocal(scratch);
        marker.position.copy(scratch);
        marker.visible = true;
        const material = marker.material as MeshStandardMaterial;
        material.emissiveIntensity = 0.35 + intensity * 0.75;
      });
    }
  });

  return (
    <group ref={groupRef} scale={layout.scale} position={[0, -0.62, 0]}>
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

export default function HandModel({ action, finger, confidence }: Props) {
  const url = "/models/hand.glb";
  const available = useModelAvailable(url);

  if (available === false) return <ProceduralHand action={action} finger={finger} confidence={confidence} />;
  if (available === null) return <ProceduralHand action={action} finger={finger} confidence={confidence} />;

  return (
    <GltfFallbackBoundary
      fallback={<ProceduralHand action={action} finger={finger} confidence={confidence} />}
      resetKey={url}
    >
      <GLTFHand action={action} finger={finger} confidence={confidence} url={url} />
    </GltfFallbackBoundary>
  );
}