/**
 * useMaskEditor — all state and raster operations for manual correction of the
 * AI-predicted segmentation mask (redesign §4).
 *
 * Design notes
 * ------------
 * • The authoritative bitmap is an OFFSCREEN canvas, not the canvas the user
 *   sees. The visible canvas is unmounted whenever the viewer switches modes
 *   (single / wipe / side-by-side), and edits must survive that; keeping the
 *   document offscreen decouples the pixels from the DOM. The visible canvas
 *   simply blits from it whenever `version` bumps.
 * • One such document per `sessionKey` (a batch case id), cached in a Map, so
 *   switching result tabs preserves each case's edits and undo stack instead of
 *   silently discarding them.
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
import type { DatasetId, MaskLayer } from '@/api/contract';
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

/** How many undo steps are retained per case. */
const HISTORY_DEPTH = 30;

/** Flood-fill colour-match tolerance, per channel (0-255). */
const FILL_TOLERANCE = 32;

/** Stable empty default so a caller passing `result?.masks ?? EMPTY` is referentially stable. */
export const NO_MASKS: MaskLayer[] = [];

/**
 * Per-dataset overlay defaults.
 *
 * CAMUS is B-mode ultrasound: dense bright speckle across the whole frame. A
 * `screen`-blended overlay lightens toward white wherever the base is already
 * bright, so on speckle the mask washes out into the noise and its boundary
 * stops reading. CAMUS therefore composites normally at high opacity. BRISC
 * (T1 MRI) has large dark regions where `screen` reads cleanly, and its tumour
 * masks are already legible, so it keeps the original treatment.
 */
const DATASET_OVERLAY: Record<DatasetId, { opacity: number; blend: 'normal' | 'screen' }> = {
  camus: { opacity: 0.95, blend: 'normal' },
  brisc: { opacity: 0.75, blend: 'screen' },
};

/** One case's editing document, cached so tab switches don't lose work. */
interface EditorSession {
  canvas: HTMLCanvasElement;
  /** The AI prediction, for "reset to AI prediction". */
  baseline: ImageData | null;
  history: ImageData[];
  index: number;
  /** False until the predicted masks have been composited in. */
  loaded: boolean;
}

/** Everything the toolkit panel and the canvas need. */
export interface MaskEditor {
  /** True once a mask has been loaded and the canvas can be edited. */
  ready: boolean;
  /** Bumps on every pixel change — consumers redraw when it changes. */
  version: number;
  /** The offscreen document canvas for the active session (null before load). */
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
  /** CSS mix-blend-mode for the overlay, chosen per dataset (see DATASET_OVERLAY). */
  blendMode: 'normal' | 'screen';
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
  /** The edited mask as a PNG data-URI, for a given session (defaults to active). */
  toDataUrl: (sessionKey?: string) => string | null;
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
 * Manual mask-editing state machine over a per-case offscreen canvas.
 *
 * @param masks      Predicted mask layers (pass a referentially stable array).
 * @param width      Mask bitmap width in px.
 * @param height     Mask bitmap height in px.
 * @param dataset    Drives the overlay opacity/blend defaults (see DATASET_OVERLAY).
 * @param sessionKey Identifies the case being edited; each key keeps its own
 *                   bitmap and undo stack, so switching batch tabs is lossless.
 */
export function useMaskEditor(
  masks: MaskLayer[],
  width: number,
  height: number,
  dataset: DatasetId,
  sessionKey: string,
): MaskEditor {
  const sessionsRef = useRef<Map<string, EditorSession>>(new Map());

  const [version, bump] = useReducer((n: number) => n + 1, 0);
  const [ready, setReady] = useState(false);
  const [history, setHistory] = useState({ canUndo: false, canRedo: false });
  const [dirty, setDirty] = useState(false);

  const [tool, setTool] = useState<EditorTool>('brush');
  const [brushSize, setBrushSize] = useState(12);
  const [eraserSize, setEraserSize] = useState(18);
  const [lassoMode, setLassoMode] = useState<LassoMode>('add');
  const [opacity, setOpacity] = useState(DATASET_OVERLAY[dataset].opacity);
  const [maskVisible, setMaskVisible] = useState(true);
  const [windowLevel, setWindowLevel] = useState(200);
  const [windowWidth, setWindowWidth] = useState(200);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });

  // Snap the overlay back to the new dataset's default when it changes — a
  // value dialled in for MRI is wrong for ultrasound speckle and vice versa.
  useEffect(() => {
    setOpacity(DATASET_OVERLAY[dataset].opacity);
  }, [dataset]);

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

  /** The cached editing session for a key, created/resized on demand. */
  const getSession = useCallback(
    (key: string): EditorSession => {
      let session = sessionsRef.current.get(key);
      if (!session) {
        session = {
          canvas: document.createElement('canvas'),
          baseline: null,
          history: [],
          index: -1,
          loaded: false,
        };
        sessionsRef.current.set(key, session);
      }
      if (session.canvas.width !== width || session.canvas.height !== height) {
        session.canvas.width = width;
        session.canvas.height = height;
      }
      return session;
    },
    [width, height],
  );

  const active = useCallback(() => getSession(sessionKey), [getSession, sessionKey]);

  const getCtx = useCallback(
    (): CanvasRenderingContext2D | null => active().canvas.getContext('2d'),
    [active],
  );

  /** Snapshot the active session's bitmap as one undo step. */
  const pushHistory = useCallback(() => {
    const session = active();
    const ctx = session.canvas.getContext('2d');
    if (!ctx) return;
    const snap = ctx.getImageData(0, 0, session.canvas.width, session.canvas.height);
    const next = session.history.slice(0, session.index + 1);
    next.push(snap);
    while (next.length > HISTORY_DEPTH) next.shift();
    session.history = next;
    session.index = next.length - 1;
    setHistory({ canUndo: next.length > 1, canRedo: false });
  }, [active]);

  // Load the predicted masks into this session, or re-attach to a session that
  // already has them (switching back to a tab must not discard its edits).
  useEffect(() => {
    let cancelled = false;
    const session = getSession(sessionKey);
    const ctx = session.canvas.getContext('2d');
    if (!ctx) return;

    if (session.loaded) {
      setReady(true);
      setHistory({
        canUndo: session.index > 0,
        canRedo: session.index < session.history.length - 1,
      });
      setDirty(session.index > 0);
      bump();
      return;
    }

    ctx.clearRect(0, 0, width, height);
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
        session.baseline = ctx.getImageData(0, 0, width, height);
        session.loaded = true;
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
  }, [masks, width, height, sessionKey, getSession, pushHistory]);

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

      // Iterative flood fill; an explicit stack avoids blowing the JS call
      // stack on large enclosed regions.
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
      const session = active();
      const ctx = session.canvas.getContext('2d');
      const snap = session.history[index];
      if (!ctx || !snap) return;
      ctx.putImageData(snap, 0, 0);
      session.index = index;
      setHistory({ canUndo: index > 0, canRedo: index < session.history.length - 1 });
      setDirty(index > 0);
      bump();
    },
    [active],
  );

  const undo = useCallback(() => {
    const session = active();
    if (session.index > 0) restore(session.index - 1);
  }, [active, restore]);

  const redo = useCallback(() => {
    const session = active();
    if (session.index < session.history.length - 1) restore(session.index + 1);
  }, [active, restore]);

  const resetToPrediction = useCallback(() => {
    const session = active();
    const ctx = session.canvas.getContext('2d');
    if (!ctx || !session.baseline) return;
    ctx.putImageData(session.baseline, 0, 0);
    bump();
    pushHistory();
    setDirty(false);
  }, [active, pushHistory]);

  const panBy = useCallback((dx: number, dy: number) => {
    setPan((p) => ({ x: p.x + dx, y: p.y + dy }));
  }, []);

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const toDataUrl = useCallback(
    (key?: string): string | null => {
      const session = sessionsRef.current.get(key ?? sessionKey);
      return session?.loaded ? session.canvas.toDataURL('image/png') : null;
    },
    [sessionKey],
  );

  const exportPng = useCallback(
    (filename: string) => {
      const url = toDataUrl();
      if (!url) return;
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
    },
    [toDataUrl],
  );

  return {
    ready,
    version,
    document: sessionsRef.current.get(sessionKey)?.canvas ?? null,
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
    blendMode: DATASET_OVERLAY[dataset].blend,
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
    toDataUrl,
  };
}
