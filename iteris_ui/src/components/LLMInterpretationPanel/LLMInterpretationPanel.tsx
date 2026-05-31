/**
 * LLMInterpretationPanel — renders the 5 labelled AI interpretation sections.
 * Accepts an async iterable of text chunks for typewriter stream rendering.
 * Section headers in font-heading teal. Copy button. Disclaimer footer.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import type { InterpretationSection } from '@/api/contract';
import { motion } from '@/tokens';

/** A labelled section with accumulated text. */
export interface SectionContent {
  id: InterpretationSection;
  label: string;
  text: string;
}

const SECTION_LABELS: Record<InterpretationSection, string> = {
  'segmentation-summary': 'Segmentation Summary',
  'clinical-significance': 'Clinical Significance',
  'metric-interpretation': 'Metric Interpretation',
  'performance-analysis': 'Performance Analysis',
  'literature-references': 'Literature References',
};

const SECTION_ORDER: InterpretationSection[] = [
  'segmentation-summary',
  'clinical-significance',
  'metric-interpretation',
  'performance-analysis',
  'literature-references',
];

/** Props for the LLMInterpretationPanel component. */
export interface LLMInterpretationPanelProps {
  /**
   * Async iterable that yields text chunks. Each chunk may start with
   * `[section-id]` to switch the active section, or plain text to append.
   */
  stream?: AsyncIterable<string>;
  /**
   * Pre-populated sections (non-streaming use).
   * If `stream` is also provided, stream takes precedence.
   */
  sections?: Partial<Record<InterpretationSection, string>>;
  /** Whether the stream is still loading. */
  loading?: boolean;
  /** Additional class names. */
  className?: string;
}

const DISCLAIMER =
  'AI-generated interpretation for research use only. Not a clinical diagnosis.';

/**
 * Renders the five ITERIS interpretation sections with typewriter streaming.
 * Section content accumulates in state as chunks arrive; copy button writes
 * the full text to the clipboard.
 */
export const LLMInterpretationPanel: React.FC<LLMInterpretationPanelProps> = ({
  stream,
  sections: staticSections,
  loading = false,
  className,
}) => {
  const [sectionText, setSectionText] = useState<Partial<Record<InterpretationSection, string>>>(
    staticSections ?? {},
  );
  const [activeSection, setActiveSection] = useState<InterpretationSection>('segmentation-summary');
  const [copying, setCopying] = useState(false);
  const abortRef = useRef(false);

  // Consume the async stream
  useEffect(() => {
    if (!stream) return;
    abortRef.current = false;

    (async () => {
      let current: InterpretationSection = 'segmentation-summary';
      for await (const chunk of stream) {
        if (abortRef.current) break;
        // Check for section marker like `[segmentation-summary]`
        const match = chunk.match(/^\[([a-z-]+)\]/);
        if (match) {
          const id = match[1] as InterpretationSection;
          if (SECTION_ORDER.includes(id)) {
            current = id;
            setActiveSection(id);
          }
          const rest = chunk.slice(match[0].length);
          if (rest) {
            setSectionText((prev) => ({
              ...prev,
              [current]: (prev[current] ?? '') + rest,
            }));
          }
        } else {
          setSectionText((prev) => ({
            ...prev,
            [current]: (prev[current] ?? '') + chunk,
          }));
        }
      }
    })();

    return () => { abortRef.current = true; };
  }, [stream]);

  // Update when static sections change (non-streaming path)
  useEffect(() => {
    if (!stream && staticSections) setSectionText(staticSections);
  }, [stream, staticSections]);

  const fullText = SECTION_ORDER.map(
    (id) => `${SECTION_LABELS[id]}\n\n${sectionText[id] ?? ''}\n`,
  ).join('\n');

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(fullText);
      setCopying(true);
      setTimeout(() => setCopying(false), motion.panelMs * 10);
    } catch {
      /* Clipboard unavailable */
    }
  }, [fullText]);

  return (
    <div
      className={[
        'bg-surface border border-border rounded-xl flex flex-col',
        className ?? '',
      ].join(' ')}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="text-sm font-heading font-semibold text-accent">AI Interpretation</h2>
        <button
          type="button"
          onClick={handleCopy}
          aria-label="Copy interpretation to clipboard"
          className="flex items-center gap-1.5 text-xs font-body text-muted hover:text-text transition-colors duration-panel ease-out"
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <rect x="4" y="1" width="7" height="8" rx="1" />
            <path d="M1 4h2m-2 0v6a1 1 0 0 0 1 1h5V9" />
          </svg>
          {copying ? 'Copied!' : 'Copy'}
        </button>
      </div>

      {/* Sections */}
      <div className="flex-1 overflow-y-auto px-4 py-3 flex flex-col gap-4">
        {loading && !Object.values(sectionText).some(Boolean) && (
          <div className="flex items-center gap-2 text-muted text-sm font-body py-4">
            {/* Pulsing dots — CSS-only, honoured by prefers-reduced-motion via global rule */}
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
            <span>Generating interpretation…</span>
          </div>
        )}

        {SECTION_ORDER.map((id) => {
          const text = sectionText[id];
          const isActive = id === activeSection && loading;
          return (
            <section key={id} aria-label={SECTION_LABELS[id]}>
              <h3 className="text-xs font-heading font-semibold text-accent uppercase tracking-wide mb-1">
                {SECTION_LABELS[id]}
              </h3>
              {text ? (
                <p className="text-sm font-body text-text whitespace-pre-wrap leading-relaxed">
                  {text}
                  {isActive && (
                    <span
                      aria-hidden="true"
                      className="inline-block w-0.5 h-3.5 ml-0.5 bg-accent align-text-bottom animate-pulse"
                    />
                  )}
                </p>
              ) : (
                <p className="text-xs font-body text-muted italic">
                  {loading ? 'Pending…' : 'No content.'}
                </p>
              )}
            </section>
          );
        })}
      </div>

      {/* Disclaimer */}
      <div className="px-4 py-2 border-t border-border">
        <p className="text-xs font-body text-muted italic">{DISCLAIMER}</p>
      </div>
    </div>
  );
};

export default LLMInterpretationPanel;
