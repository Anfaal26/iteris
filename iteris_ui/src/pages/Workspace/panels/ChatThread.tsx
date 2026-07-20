/**
 * ChatThread — the "Ask about this result" conversation UI (redesign).
 *
 * A real threaded exchange (not a one-shot generated paragraph): suggested
 * question chips above a text input, messages rendered as bubbles, the
 * assistant's latest turn streaming in live. Rendered inline at the bottom of the
 * workspace's scrollable section; the thread state itself is lifted into Workspace.
 */

import React, { useState, useRef, useEffect } from 'react';
import type { ChatMessage } from '@/api/contract';

export interface ChatThreadProps {
  messages: ChatMessage[];
  /** True while the assistant's last message is still streaming. */
  streaming: boolean;
  /** No result yet — input disabled, explanatory placeholder shown. */
  disabled: boolean;
  suggestions: string[];
  onSend: (text: string) => void;
  /** Message from the last failed turn (free-tier rate limits, outages, …). */
  error?: string | null;
  /** Re-sends the failed question. */
  onRetry?: () => void;
}

export const ChatThread: React.FC<ChatThreadProps> = ({
  messages,
  streaming,
  disabled,
  suggestions,
  onSend,
  error,
  onRetry,
}) => {
  const [draft, setDraft] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, streaming]);

  const submit = (text: string) => {
    const t = text.trim();
    if (!t || disabled || streaming) return;
    onSend(t);
    setDraft('');
  };

  const empty = messages.length === 0;

  return (
    <div className="flex flex-col">
      {/* Message list */}
      <div
        ref={scrollRef}
        className="flex flex-col gap-3 overflow-y-auto max-h-[420px]"
      >
        {empty && (
          <p className="text-sm font-body text-muted">
            {disabled
              ? 'Run inference, then ask about the boundaries, the baseline delta, or the refinement steps.'
              : 'Ask about this result — grounded in the current image, masks, and metrics.'}
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={['flex', m.role === 'user' ? 'justify-end' : 'justify-start'].join(' ')}
          >
            <div
              className={[
                'max-w-[85%] rounded-xl px-3 py-2 text-sm font-body whitespace-pre-wrap',
                m.role === 'user'
                  ? 'bg-accent/15 text-text border border-accent/30'
                  : 'bg-surface-2 text-text border border-border',
              ].join(' ')}
            >
              {m.content}
              {m.role === 'assistant' && streaming && i === messages.length - 1 && (
                <span className="inline-block w-1.5 h-3.5 ml-0.5 align-middle bg-accent animate-pulse" aria-hidden="true" />
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Inline failure + retry — the chat backend is a free tier, so a 429 or a
          cold outage is an expected outcome, not an exceptional one. */}
      {error && (
        <div
          role="alert"
          className="mt-3 flex items-start gap-2 rounded-lg border border-error/40 bg-error/10 px-3 py-2"
        >
          <span className="flex-1 text-[11px] font-body text-text">{error}</span>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="text-[11px] font-body font-semibold text-accent hover:underline flex-shrink-0"
            >
              Retry
            </button>
          )}
        </div>
      )}

      {/* Suggested-question chips (shown until the thread gets going) */}
      {!disabled && messages.length < 2 && (
        <div className="flex flex-wrap gap-1.5 mt-3">
          {suggestions.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => submit(s)}
              className="text-[11px] px-2.5 py-1 rounded-full border border-border text-muted hover:border-accent/50 hover:text-text transition-colors duration-panel ease-out"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Composer — sits in its own stacking context so the teal bloom can be
          painted behind it without escaping into the surrounding layout. */}
      <div className="relative isolate mt-3">
        {/* Ambient glow cast by the composer. Purely decorative and
            pointer-transparent; two offset radials read as light spilling out
            rather than a uniform halo. Suppressed while the input is disabled —
            nothing is glowing before there's a result to ask about. */}
        {!disabled && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute -inset-x-10 -inset-y-8 -z-10"
            style={{
              background: [
                'radial-gradient(60% 120% at 22% 50%, rgb(var(--color-chat-glow-rgb) / 0.20), transparent 70%)',
                'radial-gradient(55% 120% at 78% 50%, rgb(var(--color-chat-glow-rgb) / 0.14), transparent 70%)',
                'radial-gradient(90% 140% at 50% 50%, rgb(var(--color-chat-glow-rgb) / 0.10), transparent 75%)',
              ].join(', '),
              filter: 'blur(26px)',
            }}
          />
        )}
        <form
          className="relative flex items-center gap-2"
          onSubmit={(e) => { e.preventDefault(); submit(draft); }}
        >
          <input
            type="text"
            value={draft}
            disabled={disabled}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={disabled ? 'Run inference to ask…' : 'Ask a question…'}
            aria-label="Ask about this result"
            className="flex-1 rounded-full bg-surface-2 border border-border px-4 py-2.5 text-sm font-body text-text placeholder:text-muted/60 focus:outline-none focus:border-accent/50 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={disabled || streaming || !draft.trim()}
            aria-label="Send message"
            className="w-10 h-10 rounded-full bg-accent text-white flex items-center justify-center flex-shrink-0 disabled:bg-accent/40 disabled:cursor-not-allowed transition-colors duration-panel ease-out"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" fill="currentColor" stroke="none" />
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatThread;
