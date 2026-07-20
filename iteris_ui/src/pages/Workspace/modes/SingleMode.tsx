/**
 * SingleMode — single model viewer with canvas placeholder and mask overlays.
 *
 * Two rendering paths for the mask:
 *   • no result yet (or no editor)  → the predicted mask layers as <img> overlays
 *   • result present + editor ready → an editable <canvas> blitted from the
 *     editor's offscreen document, with pointer tools (brush / eraser / lasso /
 *     fill / pan) and a live brush-radius cursor. See hooks/useMaskEditor.
 *
 * Zoom/pan is a CSS transform on the square stack, so the canvas' own bitmap
 * resolution is untouched and pointer→document mapping stays a simple
 * getBoundingClientRect() ratio regardless of zoom.
 */

import React, { useEffect, useRef, useState } from 'react';
import type { MaskLayer } from '@/api/contract';
import type { MaskEditor, Point } from '../hooks/useMaskEditor';

/** Props for SingleMode. */
export interface SingleModeProps {
  anatomyLabel: string;
  imageB64?: string;
  masks: MaskLayer[];
  visibleStructures: Set<string>;
  overlayOpacity: number;
  windowLevel: number;
  windowWidth: number;
  /** Manual mask-editing state; when `editing` is true the canvas path is used. */
  editor?: MaskEditor;
  editing?: boolean;
}

/** Cursor keyword for each tool. */
const TOOL_CURSOR: Record<string, string> = {
  brush: 'none',
  eraser: 'none',
  lasso: 'crosshair',
  fill: 'crosshair',
  pan: 'grab',
};

/**
 * Single-model canvas view. Renders a grey placeholder with anatomy label,
 * then overlays mask images (or the editable mask canvas) from inference results.
 */
export const SingleMode: React.FC<SingleModeProps> = ({
  anatomyLabel,
  imageB64,
  masks,
  visibleStructures,
  overlayOpacity,
  windowLevel,
  windowWidth,
  editor,
  editing = false,
}) => {
  // Simulated W/L: maps to brightness/contrast CSS filter
  const brightness = 0.6 + (windowLevel / 400) * 0.8;
  const contrast = 0.8 + (windowWidth / 400) * 0.4;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastPointRef = useRef<Point | null>(null);
  const panOriginRef = useRef<{ x: number; y: number } | null>(null);
  /**
   * Lasso points live in a ref, mirrored to state only for the preview polyline.
   * Pointer events can arrive faster than React flushes state, and a functional
   * setState reading its own not-yet-committed value would drop points.
   */
  const lassoRef = useRef<Point[]>([]);
  const [lasso, setLasso] = useState<Point[]>([]);
  /** Pointer position in document coords, for the brush-radius cursor ring. */
  const [cursor, setCursor] = useState<Point | null>(null);

  const active = editing && !!editor?.ready;
  const doc = editor?.document ?? null;
  const version = editor?.version ?? 0;

  // Blit the offscreen document onto the visible canvas on every pixel change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !doc) return;
    if (canvas.width !== doc.width || canvas.height !== doc.height) {
      canvas.width = doc.width;
      canvas.height = doc.height;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(doc, 0, 0);
  }, [doc, version, active]);

  /** Client coords → document (mask bitmap) coords. */
  const toDoc = (e: React.PointerEvent): Point | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    if (!active || !editor) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    if (editor.tool === 'pan') {
      panOriginRef.current = { x: e.clientX, y: e.clientY };
      return;
    }
    const p = toDoc(e);
    if (!p) return;
    if (editor.tool === 'fill') {
      editor.fillAt(p);
      return;
    }
    if (editor.tool === 'lasso') {
      lassoRef.current = [p];
      setLasso(lassoRef.current);
      return;
    }
    lastPointRef.current = p;
    editor.drawSegment(p, p);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!active || !editor) return;
    const p = toDoc(e);
    if (p) setCursor(p);

    if (editor.tool === 'pan') {
      const origin = panOriginRef.current;
      if (!origin) return;
      editor.panBy(e.clientX - origin.x, e.clientY - origin.y);
      panOriginRef.current = { x: e.clientX, y: e.clientY };
      return;
    }
    if (!p) return;
    if (editor.tool === 'lasso') {
      if (!lassoRef.current.length) return;
      lassoRef.current = [...lassoRef.current, p];
      setLasso(lassoRef.current);
      return;
    }
    const last = lastPointRef.current;
    if (!last) return;
    editor.drawSegment(last, p);
    lastPointRef.current = p;
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (!active || !editor) return;
    e.currentTarget.releasePointerCapture(e.pointerId);
    if (editor.tool === 'pan') {
      panOriginRef.current = null;
      return;
    }
    if (editor.tool === 'lasso') {
      if (lassoRef.current.length >= 3) editor.applyLasso(lassoRef.current);
      lassoRef.current = [];
      setLasso([]);
      return;
    }
    if (lastPointRef.current) {
      lastPointRef.current = null;
      editor.commit();
    }
  };

  const transform = editor
    ? `translate(${editor.pan.x}px, ${editor.pan.y}px) scale(${editor.zoom})`
    : undefined;

  return (
    <div className="relative w-full h-full flex items-center justify-center bg-landing-bg overflow-hidden">
      {/* Canvas placeholder */}
      <div
        className="relative aspect-square"
        style={{
          width: 'min(85%, 85vh, 640px)',
          filter: `brightness(${brightness}) contrast(${contrast})`,
          transform,
        }}
      >
        {/* Base image — real scan if available, grey placeholder otherwise */}
        {imageB64 ? (
          <img
            src={imageB64}
            alt={anatomyLabel}
            className="absolute inset-0 w-full h-full object-fill rounded-lg bg-surface-2"
          />
        ) : (
          <svg
            width="100%"
            height="100%"
            viewBox="0 0 256 256"
            aria-label={`${anatomyLabel} scan placeholder`}
            role="img"
            className="rounded-lg"
          >
            <rect width="256" height="256" fill="var(--surface-2)" rx="8" />
            {[64, 128, 192].map((v) => (
              <React.Fragment key={v}>
                <line x1={v} y1="0" x2={v} y2="256" stroke="var(--border)" strokeWidth="0.5" />
                <line x1="0" y1={v} x2="256" y2={v} stroke="var(--border)" strokeWidth="0.5" />
              </React.Fragment>
            ))}
            <text
              x="128"
              y="128"
              textAnchor="middle"
              dominantBaseline="middle"
              fill="var(--muted)"
              fontSize="12"
              fontFamily="var(--font-body)"
            >
              {anatomyLabel}
            </text>
          </svg>
        )}

        {/* Mask — read-only overlays, or the editable canvas once editing is live */}
        {active ? (
          <>
            <canvas
              ref={canvasRef}
              aria-label="Editable segmentation mask"
              className="absolute inset-0 w-full h-full rounded-lg touch-none"
              style={{
                opacity: editor?.maskVisible ? editor.opacity : 0,
                mixBlendMode: 'screen',
                cursor: TOOL_CURSOR[editor?.tool ?? 'brush'] ?? 'crosshair',
              }}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={() => setCursor(null)}
            />

            {/* Lasso preview + live brush radius, both in document coordinates */}
            {editor && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox={`0 0 ${doc?.width ?? 256} ${doc?.height ?? 256}`}
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                {lasso.length > 1 && (
                  <polyline
                    points={lasso.map((p) => `${p.x},${p.y}`).join(' ')}
                    fill="none"
                    stroke="var(--color-accent)"
                    strokeWidth="1.5"
                    strokeDasharray="4 3"
                  />
                )}
                {cursor && (editor.tool === 'brush' || editor.tool === 'eraser') && (
                  <circle
                    cx={cursor.x}
                    cy={cursor.y}
                    r={editor.activeSize / 2}
                    fill="none"
                    stroke={editor.tool === 'brush' ? editor.activeColor : 'var(--color-error)'}
                    strokeWidth="1"
                  />
                )}
              </svg>
            )}
          </>
        ) : (
          masks.map((mask) =>
            visibleStructures.has(mask.structure) ? (
              <img
                key={mask.structure}
                src={mask.imageB64}
                alt={`${mask.label} segmentation mask`}
                className="absolute inset-0 w-full h-full rounded-lg"
                style={{ opacity: overlayOpacity, mixBlendMode: 'screen' }}
              />
            ) : null,
          )
        )}
      </div>
    </div>
  );
};

export default SingleMode;
