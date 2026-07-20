/**
 * useMaskEditor — all state and raster operations for manual correction of the
 * AI-predicted segmentation mask (redesign §4).
 *
 * Design notes
 * ------------
 * • The authoritative bitmap is an OFFSCREEN canvas held in a ref, not the
 *   canvas the user sees. The visible canvas is unmounted whenever the viewer
 *   switches modes (single / wipe / side-by-side), and edits must survive that;
 *   keeping the document offscreen decouples the pixels from the DOM. The
 *   visible canvas simply blits from it whenever `version` bumps.
 * • The mask predicted by the backend arrives as one PNG/SVG data-URI per
 *   structure. They are composited onto that single document canvas on load, so
 *   the editor is a painting surface rather than a per-class stack — painting
 *   picks a class *colour* from the same reserved mask palette (src/index.css
 *   --mask-*), which is what the overlay renders anyway.
 * • Undo/redo is snapshot-based (ImageData, depth HISTORY_DEPTH). Snapshots are
 *   taken at stroke *end*, so one drag = one undo step.
 *
 * State lives here and is lifted into Workspace so the right-hand toolkit panel
 * and the centre canvas drive the same instance — the existing props-down
 * pattern, no store or context introduced.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';
import type { MaskLayer } from '@/api/contract';
import { structureColorHex } from '@/tokens';

/** Active editing tool. */
export type EditorTool = 'brush' | 'eraser' | 'lasso' | 'fill' | 'pan';

/** Whether the lasso adds to or removes from the mask. */
export type LassoMode = 'add' | 'remove';

/** A selectable paint colour, mirroring one segmentation class. */
export interface PaletteEntry {
  structure: string;
  label: string;
  color: string;
}

/** A point in document (mask bitmap) coordinates. */
export interface Point {
  x: number;
  y: number;
}

/** How many undo steps are retained. */
const HISTORY_DEPTH = 30;

/** Flood-fill colour-match tolerance, per channel (0-255). */
const FILL_TOLERANCE = 32;

/** Stable empty default so a caller passing `result?.masks ?? EMPTY` is referentially stable. */
export const NO_MASKS: MaskLayer[] = [];

/** Everything the toolkit panel and the canvas need. */
export interface MaskEditor {
  /** True once a mask has been loaded and the canvas can be edited. */
  ready: boolean;
  /** Bumps on every pixel change — consumers redraw when it changes. */
  version: number;
  /** The offscreen document canvas (null before the first mask loads). */
  document: HTMLCanvasElement | null;

  tool: EditorTool;
  setTool: (t: EditorTool) => void;
  brushSize: number;
  setBrushSize: (n: number) => void;
  eraserSize: number;
  setEraserSize: (n: number) => void;
  /** Radius in document px of whichever size-bearing tool is active. */
  activeSize: number;

  palette: PaletteEntry[];
  activeColor: string;
  setActiveColor: (c: string) => void;

  lassoMode: LassoMode;
  setLassoMode: (m: LassoMode) => void;

  opacity: number;
  setOpacity: (n: number) => void;
  maskVisible: boolean;
  setMaskVisible: (v: boolean) => void;

  /** Window/level for the underlying scan (radiology-style brightness/contrast). */
  windowLevel: number;
  setWindowLevel: (n: number) => void;
  windowWidth: number;
  setWindowWidth: (n: number) => void;

  zoom: number;
  setZoom: (n: number) => void;
  pan: Point;
  setPan: (p: Point) => void;
  panBy: (dx: number, dy: number) => void;
  resetView: () => void;

  canUndo: boolean;
  canRedo: boolean;
  undo: () => void;
  redo: () => void;
  /** True when the mask differs from the AI prediction. */
  dirty: boolean;
  resetToPrediction: () => void;

  /* --- raster operations, called by the canvas --- */
  /** Paint (or erase) a segment of a freehand stroke. */
  drawSegment: (from: Point, to: Point) => void;
  /** Commit the current stroke as one undo step. */
  commit: () => void;
  /** Flood-fill the contiguous region under `p`. */
  fillAt: (p: Point) => void;
  /** Apply a closed lasso polygon (add or remove per `lassoMode`). */
  applyLasso: (points: Point[]) => void;

  /** Download the edited mask as a PNG. */
  exportPng: (filename: string) => void;
}

/** Loads a data-URI / URL into an HTMLImageElement. */
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load mask image`));
    img.src = src.startsWith('data:') || src.startsWith('http') ? src : `data:image/png;base64,${src}`;
  });
}

/** #rrggbb → [r,g,b]. */
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  return [
    parseInt(full.slice(0, 2), 16),
    parseInt(full.slice(2, 4), 16),
    parseInt(full.slice(4, 6), 16),
  ];
}

/**
 * Manual mask-editing state machine over an offscreen canvas.
 *
 * @param masks  Predicted mask layers (pass a referentially stable array).
 * @param width  Mask bitmap width in px.
 * @param height Mask bitmap height in px.
 */
export function useMaskEditor(masks: MaskLayer[], width: number, height: number): MaskEditor {
  const docRef = useRef<HTMLCanvasElement | null>(null);
  const baselineRef = useRef<ImageData | null>(null);
  const historyRef = useRef<ImageData[]>([]);
  const indexRef = useRef(-1);

  const [version, bump] = useReducer((n: number) => n + 1, 0);
  const [ready, setReady] = useState(false);
  const [history, setHistory] = useState({ canUndo: false, canRedo: false });
  const [dirty, setDirty] = useState(false);

  const [tool, setTool] = useState<EditorTool>('brush');
  const [brushSize, setBrushSize] = useState(12);
  const [eraserSize, setEraserSize] = useState(18);
  const [lassoMode, setLassoMode] = useState<LassoMode>('add');
  const [opacity, setOpacity] = useState(0.75);
  const [maskVisible, setMaskVisible] = useState(true);
  const [windowLevel, setWindowLevel] = useState(200);
  const [windowWidth, setWindowWidth] = useState(200);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });

  const palette = useMemo<PaletteEntry[]>(
    () =>
      masks.map((m) => ({
        structure: m.structure,
        label: m.label,
        // Backend/mock colours are hex; fall back to the reserved palette token.
        color: m.color?.startsWith('#') ? m.color : (structureColorHex[m.structure] ?? '#3b82f6'),
      })),
    [masks],
  );

  const [activeColor, setActiveColor] = useState<string>('');
  // Keep the paint colour valid whenever the class palette changes (new dataset).
  useEffect(() => {
    setActiveColor((current) =>
      palette.some((p) => p.color === current) ? current : (palette[0]?.color ?? '#3b82f6'),
    );
  }, [palette]);

  /** The offscreen document canvas, created/resized on demand. */
  const getDoc = useCallback((): HTMLCanvasElement => {
    let canvas = docRef.current;
    if (!canvas) {
      canvas = document.createElement('canvas');
      docRef.current = canvas;
    }
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return canvas;
  }, [width, height]);

  const getCtx = useCallback((): CanvasRenderingContext2D | null => getDoc().getContext('2d'), [getDoc]);

  /** Snapshot the current bitmap as one undo step. */
  const pushHistory = useCallback(() => {
    const ctx = getCtx();
    if (!ctx) return;
    const snap = ctx.getImageData(0, 0, width, height);
    const next = historyRef.current.slice(0, indexRef.current + 1);
    next.push(snap);
    while (next.length > HISTORY_DEPTH) next.shift();
    historyRef.current = next;
    indexRef.current = next.length - 1;
    setHistory({ canUndo: next.length > 1, canRedo: false });
  }, [getCtx, width, height]);

  // Load the predicted masks onto the document canvas whenever they change.
  useEffect(() => {
    let cancelled = false;
    const ctx = getCtx();
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    historyRef.current = [];
    indexRef.current = -1;
    baselineRef.current = null;
    setHistory({ canUndo: false, canRedo: false });
    setDirty(false);

    if (masks.length === 0) {
      setReady(false);
      bump();
      return;
    }

    Promise.all(masks.map((m) => loadImage(m.imageB64)))
      .then((images) => {
        if (cancelled) return;
        images.forEach((img) => ctx.drawImage(img, 0, 0, width, height));
        baselineRef.current = ctx.getImageData(0, 0, width, height);
        setReady(true);
        pushHistory();
        bump();
      })
      .catch(() => {
        if (!cancelled) setReady(false);
      });

    return () => {
      cancelled = true;
    };
  }, [masks, width, height, getCtx, pushHistory]);

  const drawSegment = useCallback(
    (from: Point, to: Point) => {
      const ctx = getCtx();
      if (!ctx || (tool !== 'brush' && tool !== 'eraser')) return;
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.lineWidth = tool === 'brush' ? brushSize : eraserSize;
      ctx.globalCompositeOperation = tool === 'brush' ? 'source-over' : 'destination-out';
      ctx.strokeStyle = tool === 'brush' ? activeColor : 'rgba(0,0,0,1)';
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
      ctx.restore();
      bump();
    },
    [getCtx, tool, brushSize, eraserSize, activeColor],
  );

  const commit = useCallback(() => {
    pushHistory();
    setDirty(true);
  }, [pushHistory]);

  const applyLasso = useCallback(
    (points: Point[]) => {
      const ctx = getCtx();
      if (!ctx || points.length < 3) return;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.closePath();
      ctx.globalCompositeOperation = lassoMode === 'add' ? 'source-over' : 'destination-out';
      ctx.fillStyle = lassoMode === 'add' ? activeColor : 'rgba(0,0,0,1)';
      ctx.fill();
      ctx.restore();
      bump();
      pushHistory();
      setDirty(true);
    },
    [getCtx, lassoMode, activeColor, pushHistory],
  );

  const fillAt = useCallback(
    (p: Point) => {
      const ctx = getCtx();
      if (!ctx) return;
      const x0 = Math.round(p.x);
      const y0 = Math.round(p.y);
      if (x0 < 0 || y0 < 0 || x0 >= width || y0 >= height) return;

      const image = ctx.getImageData(0, 0, width, height);
      const data = image.data;
      const at = (x: number, y: number) => (y * width + x) * 4;
      const seed = at(x0, y0);
      const target = [data[seed], data[seed + 1], data[seed + 2], data[seed + 3]];
      const [fr, fg, fb] = hexToRgb(activeColor);
      // Already the fill colour and opaque — nothing to do.
      if (target[3] === 255 && target[0] === fr && target[1] === fg && target[2] === fb) return;

      const matches = (i: number) =>
        Math.abs(data[i] - target[0]) <= FILL_TOLERANCE &&
        Math.abs(data[i + 1] - target[1]) <= FILL_TOLERANCE &&
        Math.abs(data[i + 2] - target[2]) <= FILL_TOLERANCE &&
        Math.abs(data[i + 3] - target[3]) <= FILL_TOLERANCE;

      // Iterative scanline-free flood fill; an explicit stack avoids blowing the
      // JS call stack on large enclosed regions.
      const stack: number[] = [x0, y0];
      const seen = new Uint8Array(width * height);
      seen[y0 * width + x0] = 1;
      while (stack.length) {
        const y = stack.pop() as number;
        const x = stack.pop() as number;
        const i = at(x, y);
        data[i] = fr;
        data[i + 1] = fg;
        data[i + 2] = fb;
        data[i + 3] = 255;
        const neighbours: [number, number][] = [
          [x + 1, y],
          [x - 1, y],
          [x, y + 1],
          [x, y - 1],
        ];
        for (const [nx, ny] of neighbours) {
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const key = ny * width + nx;
          if (seen[key]) continue;
          if (!matches(at(nx, ny))) continue;
          seen[key] = 1;
          stack.push(nx, ny);
        }
      }
      ctx.putImageData(image, 0, 0);
      bump();
      pushHistory();
      setDirty(true);
    },
    [getCtx, width, height, activeColor, pushHistory],
  );

  /** Restore a history snapshot at `index`. */
  const restore = useCallback(
    (index: number) => {
      const ctx = getCtx();
      const snap = historyRef.current[index];
      if (!ctx || !snap) return;
      ctx.putImageData(snap, 0, 0);
      indexRef.current = index;
      setHistory({ canUndo: index > 0, canRedo: index < historyRef.current.length - 1 });
      setDirty(index > 0);
      bump();
    },
    [getCtx],
  );

  const undo = useCallback(() => {
    if (indexRef.current > 0) restore(indexRef.current - 1);
  }, [restore]);

  const redo = useCallback(() => {
    if (indexRef.current < historyRef.current.length - 1) restore(indexRef.current + 1);
  }, [restore]);

  const resetToPrediction = useCallback(() => {
    const ctx = getCtx();
    const base = baselineRef.current;
    if (!ctx || !base) return;
    ctx.putImageData(base, 0, 0);
    bump();
    pushHistory();
    setDirty(false);
  }, [getCtx, pushHistory]);

  const panBy = useCallback((dx: number, dy: number) => {
    setPan((p) => ({ x: p.x + dx, y: p.y + dy }));
  }, []);

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const exportPng = useCallback(
    (filename: string) => {
      const canvas = docRef.current;
      if (!canvas) return;
      const url = canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
    },
    [],
  );

  return {
    ready,
    version,
    document: docRef.current,
    tool,
    setTool,
    brushSize,
    setBrushSize,
    eraserSize,
    setEraserSize,
    activeSize: tool === 'eraser' ? eraserSize : brushSize,
    palette,
    activeColor,
    setActiveColor,
    lassoMode,
    setLassoMode,
    opacity,
    setOpacity,
    maskVisible,
    setMaskVisible,
    windowLevel,
    setWindowLevel,
    windowWidth,
    setWindowWidth,
    zoom,
    setZoom,
    pan,
    setPan,
    panBy,
    resetView,
    canUndo: history.canUndo,
    canRedo: history.canRedo,
    undo,
    redo,
    dirty,
    resetToPrediction,
    drawSegment,
    commit,
    fillAt,
    applyLasso,
    exportPng,
  };
}
