import React, { useEffect, useRef } from "react";
import * as THREE from "three";

export type GraphNode = {
  id: string;
  title: string;
  kind: string;
  params?: number;
  macs?: number;
  shape?: string;
  shape_in?: string;
  shape_out?: string;
};

export type GraphEdge = { from: string; to: string };

export type Graph3DProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  grouping?: boolean;
  selectedId?: string | null;
  onSelect?: (id: string | null) => void;
};

const ORDER = [
  "input",
  "conv1",
  "gn1",
  "relu1",
  "drop1",
  "conv2",
  "gn2",
  "relu2",
  "drop2",
  "lstm",
  "last",
  "head_dropout",
  "finger_head",
  "action_head"
];

function getColor(kind: string, grouping: boolean): number {
  if (!grouping) {
    return 0x60a5fa;
  }
  if (kind === "input") return 0x38bdf8;
  if (kind === "conv1d" || kind === "norm" || kind === "activation" || kind === "dropout") return 0xa855f7;
  if (kind === "lstm" || kind === "pool") return 0xf97316;
  if (kind === "linear") return 0x22c55e;
  return 0x94a3b8;
}

function createLabelSprite(text: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return new THREE.Sprite();
  }
  canvas.width = 256;
  canvas.height = 64;
  ctx.fillStyle = "rgba(15,23,42,0.9)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f8fafc";
  ctx.font = "18px 'Avenir Next', sans-serif";
  ctx.fillText(text, 12, 38);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(2.2, 0.6, 1);
  return sprite;
}

export default function Graph3D({ nodes, edges, grouping = true, selectedId, onSelect }: Graph3DProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const width = container.clientWidth || 600;
    const height = container.clientHeight || 320;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0e10);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 0, 20);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambient);

    const positions: Record<string, THREE.Vector3> = {};
    const spacing = 2.2;
    ORDER.forEach((id, idx) => {
      const x = (idx - ORDER.length / 2) * spacing;
      let y = 0;
      if (id === "finger_head") y = 1.6;
      if (id === "action_head") y = -1.6;
      positions[id] = new THREE.Vector3(x, y, 0);
    });

    const nodeMeshes: THREE.Mesh[] = [];
    nodes.forEach((node) => {
      const pos = positions[node.id] ?? new THREE.Vector3(0, 0, 0);
      const geometry = new THREE.BoxGeometry(1.0, 0.8, 0.4);
      const color = getColor(node.kind, grouping);
      const material = new THREE.MeshStandardMaterial({ color });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(pos);
      mesh.userData = { id: node.id };
      scene.add(mesh);
      nodeMeshes.push(mesh);

      const label = createLabelSprite(node.title);
      label.position.copy(pos.clone().add(new THREE.Vector3(0, 0.8, 0)));
      scene.add(label);
    });

    edges.forEach((edge) => {
      const from = positions[edge.from];
      const to = positions[edge.to];
      if (!from || !to) return;
      const points = [from.clone(), to.clone()];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: 0x94a3b8 });
      const line = new THREE.Line(geometry, material);
      scene.add(line);
    });

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    function handlePointer(event: MouseEvent) {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(nodeMeshes, false);
      if (hits.length > 0) {
        const id = hits[0].object.userData.id as string;
        onSelect?.(id);
      } else {
        onSelect?.(null);
      }
    }

    renderer.domElement.addEventListener("click", handlePointer);

    function render() {
      nodeMeshes.forEach((mesh) => {
        const id = mesh.userData.id as string;
        const material = mesh.material as THREE.MeshStandardMaterial;
        const base = getColor(nodes.find((n) => n.id === id)?.kind ?? "", grouping);
        material.color.setHex(id === selectedId ? 0xfacc15 : base);
      });
      renderer.render(scene, camera);
    }

    render();

    return () => {
      renderer.domElement.removeEventListener("click", handlePointer);
      renderer.dispose();
    };
  }, [nodes, edges, grouping, selectedId, onSelect]);

  return <div className="nnvis-graph" ref={containerRef} />;
}
