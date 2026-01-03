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

function extractIndexFromName(name: string): number | null {
  const match = name.match(/(\d+)(?!.*\d)/);
  if (!match) return null;
  return Number.parseInt(match[1], 10);
}

function buildBoneChain(bones: Bone[]): Bone[] {
  if (bones.length <= 1) return bones.slice();
  const boneSet = new Set(bones);
  const sortKey = (bone: Bone) => {
    const idx = extractIndexFromName(bone.name);
    if (idx !== null) return idx;
    let depth = 0;
    let current: Object3D | null = bone.parent;
    while (current) {
      depth += 1;
      current = current.parent;
    }
    return depth + 10;
  };

  const roots = bones.filter((bone) => !boneSet.has(bone.parent as Bone));
  const candidates = roots.length ? roots : bones;
  const root = candidates.slice().sort((a, b) => sortKey(a) - sortKey(b))[0];
  const chain: Bone[] = [];
  const visited = new Set<Bone>();
  let current: Bone | undefined = root;

  while (current && !visited.has(current)) {
    chain.push(current);
    visited.add(current);
    const childBones: Bone[] = current.children.filter((child): child is Bone =>
      boneSet.has(child as Bone)
    );
    if (!childBones.length) break;
    childBones.sort((a, b) => sortKey(a) - sortKey(b));
    current = childBones[0];
  }

  const remaining = bones.filter((bone) => !visited.has(bone));
  remaining.sort((a, b) => sortKey(a) - sortKey(b));
  return chain.concat(remaining);
}

function segmentWeight(count: number, index: number) {
  if (count <= 1) return 0.7;
  const t = index / (count - 1);
  return MathUtils.lerp(0.8, 0.35, t);
}

function detectThumbAxis(chain: Bone[]): "x" | "z" {
  if (!chain.length) return "x";
  const base = chain[0];
  const child =
    chain[1] ??
    base.children.find((node): node is Bone => (node as Bone).isBone);
  if (!child) return "x";
  const dir = new Vector3().subVectors(child.position, base.position);
  if (dir.lengthSq() < 1e-4) return "x";
  dir.normalize();
  const sideways = Math.abs(dir.x) > Math.max(Math.abs(dir.y), Math.abs(dir.z));
  return sideways ? "z" : "x";
}

function targetFlexFromAction(action: string) {
  if (action === "CLOSE") return 1;
  if (action === "OPEN") return 0;
  return 0.35;
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

function computeFingerInfluence(
  mesh: SkinnedMesh,
  fingerBoneIndices: Record<FingerKey, Set<number>>
) {
  const skinIndex = mesh.geometry.getAttribute("skinIndex");
  const skinWeight = mesh.geometry.getAttribute("skinWeight");
  if (!skinIndex || !skinWeight) return null;

  const fingerWeights: FingerWeights = {
    THUMB: 0,
    INDEX: 0,
    MIDDLE: 0,
    RING: 0,
    PINKY: 0
  };
  let totalWeight = 0;

  const indexToFingers = new Map<number, FingerKey[]>();
  FINGER_ORDER.forEach((fingerName) => {
    fingerBoneIndices[fingerName].forEach((index) => {
      const existing = indexToFingers.get(index);
      if (existing) {
        existing.push(fingerName);
      } else {
        indexToFingers.set(index, [fingerName]);
      }
    });
  });

  for (let i = 0; i < skinIndex.count; i += 1) {
    const indices = [
      skinIndex.getX(i),
      skinIndex.getY(i),
      skinIndex.getZ(i),
      skinIndex.getW(i)
    ];
    const weights = [
      skinWeight.getX(i),
      skinWeight.getY(i),
      skinWeight.getZ(i),
      skinWeight.getW(i)
    ];

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
    const target = targetFlexFromAction(action);
    flexRef.current = MathUtils.damp(flexRef.current, target, 6, delta);

    const highlighted = normalizeFinger(finger);
    const intensity = highlighted === "NONE" ? 0 : MathUtils.clamp(confidence, 0, 1);

    FINGER_ORDER.forEach((name) => {
      const group = fingerRefs.current[name];
      const segments = segmentRefs.current[name] ?? [];
      if (group) {
        const baseRot = group.userData.baseRotation as { x: number; y: number; z: number } | undefined;
        if (!baseRot) {
          group.userData.baseRotation = { x: group.rotation.x, y: group.rotation.y, z: group.rotation.z };
        }
      }

      segments.forEach((segment, idx) => {
        if (!segment) return;
        segment.rotation.x = -flexRef.current * (0.35 + idx * 0.35);
      });

      const mat = materials[name];
      if (mat) {
        const isActive = highlighted === name;
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
      if ((obj as Mesh).isMesh) {
        allMeshes.push(obj as Mesh);
      }
      if ((obj as SkinnedMesh).isSkinnedMesh) {
        skinnedMeshes.push(obj as SkinnedMesh);
      }
      const key = fingerKeyFromName(obj.name);
      if (key) {
        fingerGroups[key].push(obj);
      }
    });

    const boneSet = new Set<Bone>();
    skinnedMeshes.forEach((mesh) => {
      mesh.skeleton?.bones.forEach((bone) => boneSet.add(bone));
    });

    const fingerBones: Record<FingerKey, Bone[]> = {
      THUMB: [],
      INDEX: [],
      MIDDLE: [],
      RING: [],
      PINKY: []
    };

    boneSet.forEach((bone) => {
      const key = fingerKeyFromName(bone.name);
      if (key) {
        fingerBones[key].push(bone);
      }
    });

    const fingerChains: Record<FingerKey, Bone[]> = {
      THUMB: buildBoneChain(fingerBones.THUMB),
      INDEX: buildBoneChain(fingerBones.INDEX),
      MIDDLE: buildBoneChain(fingerBones.MIDDLE),
      RING: buildBoneChain(fingerBones.RING),
      PINKY: buildBoneChain(fingerBones.PINKY)
    };

    const fingerAxes: Record<FingerKey, "x" | "z"> = {
      THUMB: detectThumbAxis(fingerChains.THUMB),
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

    FINGER_ORDER.forEach((name) => {
      fingerGroups[name].forEach((obj) => {
        obj.traverse((child) => {
          if ((child as Mesh).isMesh) {
            fingerMeshes[name].add(child as Mesh);
          }
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
        if (key) {
          boneIndices[key].add(index);
        }
      });

      const influence = computeFingerInfluence(mesh, boneIndices);
      if (!influence || influence.totalWeight <= 0) return;
      FINGER_ORDER.forEach((fingerName) => {
        const share = influence.fingerWeights[fingerName] / influence.totalWeight;
        if (share >= 0.35) {
          fingerMeshes[fingerName].add(mesh);
        }
      });
    });

    return {
      allMeshes,
      fingerGroups,
      fingerChains,
      fingerAxes,
      fingerTips,
      fingerMeshes,
      hasRig: boneSet.size > 0
    };
  }, [gltf]);

  const layout = useMemo(() => {
    const box = new Box3().setFromObject(gltf.scene);
    const size = new Vector3();
    const center = new Vector3();
    box.getSize(size);
    box.getCenter(center);
    // Shrink the rigged model so it reads smaller in the stage.
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const scale = MathUtils.clamp(1.2 / maxDim, 0.7, 1.6) * 0.5;
    const offset = new Vector3(-center.x, -box.min.y, -center.z);
    return { scale, offset };
  }, [gltf]);

  useEffect(() => {
    rig.allMeshes.forEach((mesh) => {
      if (Array.isArray(mesh.material)) {
        mesh.material = mesh.material.map((mat) => mat.clone()) as MeshStandardMaterial[];
      } else if (mesh.material) {
        mesh.material = (mesh.material as MeshStandardMaterial).clone();
      }
    });
  }, [rig.allMeshes]);

  useFrame((_, delta) => {
    const target = targetFlexFromAction(action);
    const intensity = MathUtils.clamp(confidence, 0, 1);
    const speed = 6 + intensity * 2;
    flexRef.current = MathUtils.damp(flexRef.current, target, speed, delta);
    const flex = flexRef.current * (1 + intensity * 0.08);

    const highlighted = normalizeFinger(finger);
    const hasBoneChains = FINGER_ORDER.some((name) => rig.fingerChains[name].length > 0);

    if (hasBoneChains) {
      FINGER_ORDER.forEach((name, idx) => {
        const chain = rig.fingerChains[name];
        if (!chain.length) return;
        const axis = rig.fingerAxes[name];
        const fingerFlex = flex * (0.88 + idx * 0.05);
        chain.forEach((bone, boneIndex) => {
          const base = bone.userData.baseRotation as { x: number; y: number; z: number } | undefined;
          if (!base) {
            bone.userData.baseRotation = { x: bone.rotation.x, y: bone.rotation.y, z: bone.rotation.z };
          }
          const saved = bone.userData.baseRotation as { x: number; y: number; z: number };
          const amount = fingerFlex * segmentWeight(chain.length, boneIndex);
          if (axis === "x") bone.rotation.x = saved.x - amount;
          if (axis === "z") bone.rotation.z = saved.z - amount;
        });
      });
    } else {
      const hasFingerGroups = FINGER_ORDER.some((name) => rig.fingerGroups[name].length > 0);
      if (hasFingerGroups) {
        FINGER_ORDER.forEach((name, idx) => {
          const axis = name === "THUMB" ? "z" : "x";
          const groupFlex = flex * (0.45 + idx * 0.1);
          rig.fingerGroups[name].forEach((obj) => {
            const base = obj.userData.baseRotation as { x: number; y: number; z: number } | undefined;
            if (!base) {
              obj.userData.baseRotation = { x: obj.rotation.x, y: obj.rotation.y, z: obj.rotation.z };
            }
            const saved = obj.userData.baseRotation as { x: number; y: number; z: number };
            if (axis === "x") obj.rotation.x = saved.x - groupFlex;
            if (axis === "z") obj.rotation.z = saved.z - groupFlex * 0.6;
          });
        });
      } else {
        const root = gltf.scene as Group;
        const base = root.userData.baseRotation as { x: number; y: number; z: number } | undefined;
        if (!base) {
          root.userData.baseRotation = { x: root.rotation.x, y: root.rotation.y, z: root.rotation.z };
        }
        const saved = root.userData.baseRotation as { x: number; y: number; z: number };
        root.rotation.x = saved.x - flex * 0.2;
      }
    }

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

  // Align model base to the ground plane so it sits on the stage bottom.
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

  if (available === false) {
    return <ProceduralHand action={action} finger={finger} confidence={confidence} />;
  }

  if (available === null) {
    return <ProceduralHand action={action} finger={finger} confidence={confidence} />;
  }

  return (
    <GltfFallbackBoundary
      fallback={<ProceduralHand action={action} finger={finger} confidence={confidence} />}
      resetKey={url}
    >
      <GLTFHand action={action} finger={finger} confidence={confidence} url={url} />
    </GltfFallbackBoundary>
  );
}
