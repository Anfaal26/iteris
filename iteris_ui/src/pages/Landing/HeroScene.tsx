/**
 * HeroScene — Three.js skeletal/neural lattice for the Landing page hero.
 *
 * ~60 nodes (icosahedron-based point cloud), ~80 edges (LineSegments),
 * slow rotation, node glow pulse, mouse parallax ±5° camera tilt.
 * Capped at 30 fps via elapsed-time gating (motion.landingFps).
 * Disposes all geometries and materials on unmount.
 */

import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { colorsHex, motion } from '@/tokens';

/** Props for HeroScene. */
export interface HeroSceneProps {
  /** Additional class names for the container div. */
  className?: string;
}

const FRAME_INTERVAL_MS = 1000 / motion.landingFps; // ~33.33 ms
const NODE_COUNT = 60;
const EDGE_COUNT = 80;
const ROTATION_SPEED = 0.003;

/** Generates a pseudo-icosahedron-like point distribution on a sphere. */
function buildNodePositions(count: number, radius: number): Float32Array {
  const positions = new Float32Array(count * 3);
  // Fibonacci sphere for even distribution
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = 1 - (i / (count - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = goldenAngle * i;
    positions[i * 3] = Math.cos(theta) * r * radius;
    positions[i * 3 + 1] = y * radius;
    positions[i * 3 + 2] = Math.sin(theta) * r * radius;
  }
  return positions;
}

/** Generates edge indices connecting nearby nodes. */
function buildEdgeIndices(positions: Float32Array, nodeCount: number, edgeCount: number): Float32Array {
  const edgePositions: number[] = [];
  const pairs: [number, number][] = [];

  // Build candidate pairs sorted by distance
  type DistPair = { i: number; j: number; d: number };
  const dists: DistPair[] = [];
  for (let i = 0; i < nodeCount; i++) {
    for (let j = i + 1; j < nodeCount; j++) {
      const dx = positions[i * 3] - positions[j * 3];
      const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
      const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
      dists.push({ i, j, d: dx * dx + dy * dy + dz * dz });
    }
  }
  dists.sort((a, b) => a.d - b.d);

  const used = dists.slice(0, edgeCount);
  for (const { i, j } of used) {
    pairs.push([i, j]);
    edgePositions.push(
      positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2],
      positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2],
    );
  }

  return new Float32Array(edgePositions);
}

/**
 * Three.js lattice scene that renders into a canvas filling its container.
 * Uses `colorsHex.gradientA` for nodes and `colorsHex.gradientB` for edges.
 */
export const HeroScene: React.FC<HeroSceneProps> = ({ className }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // ── Renderer ──────────────────────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // ── Scene & Camera ────────────────────────────────────────────────────────
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
    camera.position.set(0, 0, 4.5);

    // ── Geometry ──────────────────────────────────────────────────────────────
    const nodePositions = buildNodePositions(NODE_COUNT, 1.8);
    const edgePositions = buildEdgeIndices(nodePositions, NODE_COUNT, EDGE_COUNT);

    // Nodes
    const nodeGeo = new THREE.BufferGeometry();
    nodeGeo.setAttribute('position', new THREE.BufferAttribute(nodePositions, 3));
    const nodeMat = new THREE.PointsMaterial({
      color: new THREE.Color(colorsHex.gradientA),
      size: 0.045,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
    });
    const nodes = new THREE.Points(nodeGeo, nodeMat);
    scene.add(nodes);

    // Edges
    const edgeGeo = new THREE.BufferGeometry();
    edgeGeo.setAttribute('position', new THREE.BufferAttribute(edgePositions, 3));
    const edgeMat = new THREE.LineBasicMaterial({
      color: new THREE.Color(colorsHex.gradientB),
      transparent: true,
      opacity: 0.3,
    });
    const edges = new THREE.LineSegments(edgeGeo, edgeMat);
    scene.add(edges);

    // ── Resize handling ────────────────────────────────────────────────────────
    const resize = () => {
      const { clientWidth: w, clientHeight: h } = container;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const resizeObs = new ResizeObserver(resize);
    resizeObs.observe(container);

    // ── Mouse parallax ────────────────────────────────────────────────────────
    const mouse = { x: 0, y: 0 };
    const onMouseMove = (e: MouseEvent) => {
      mouse.x = (e.clientX / window.innerWidth - 0.5) * 2;
      mouse.y = (e.clientY / window.innerHeight - 0.5) * 2;
    };
    window.addEventListener('mousemove', onMouseMove, { passive: true });

    // ── Animation loop (30fps cap) ─────────────────────────────────────────────
    let rafId = 0;
    let lastTime = 0;
    let elapsedTime = 0;

    const animate = (now: number) => {
      rafId = requestAnimationFrame(animate);
      const delta = now - lastTime;
      if (delta < FRAME_INTERVAL_MS) return;
      lastTime = now - (delta % FRAME_INTERVAL_MS);
      elapsedTime += delta * 0.001; // seconds

      // Slow rotation
      nodes.rotation.y += ROTATION_SPEED;
      edges.rotation.y += ROTATION_SPEED;
      nodes.rotation.x += ROTATION_SPEED * 0.3;
      edges.rotation.x += ROTATION_SPEED * 0.3;

      // Node glow pulse via opacity
      nodeMat.opacity = 0.55 + 0.35 * Math.sin(elapsedTime * 1.5);

      // Camera parallax tilt (±5° = ±0.087 rad)
      const maxTilt = (5 * Math.PI) / 180;
      camera.rotation.y = THREE.MathUtils.lerp(camera.rotation.y, -mouse.x * maxTilt, 0.05);
      camera.rotation.x = THREE.MathUtils.lerp(camera.rotation.x, mouse.y * maxTilt, 0.05);

      renderer.render(scene, camera);
    };
    rafId = requestAnimationFrame(animate);

    // ── Cleanup ────────────────────────────────────────────────────────────────
    return () => {
      cancelAnimationFrame(rafId);
      resizeObs.disconnect();
      window.removeEventListener('mousemove', onMouseMove);
      nodeGeo.dispose();
      edgeGeo.dispose();
      nodeMat.dispose();
      edgeMat.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className={['w-full h-full', className ?? ''].join(' ')}
      aria-hidden="true"
    />
  );
};

export default HeroScene;
