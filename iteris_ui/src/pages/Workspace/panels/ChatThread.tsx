/**
 * ChatThread — the "Ask about this result" conversation UI (redesign).
 *
 * A real threaded exchange (not a one-shot generated paragraph): suggested
 * question chips above a text input, messages rendered as bubbles, the
 * assistant's latest turn streaming in live. Rendered both inline at the bottom
 * of the page and inside the floating chat bubble, sharing one lifted thread.
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
  /** 'inline' fills its container; 'popover' is the compact bubble variant. */
  variant?: 'inline' | 'popover';
}

export const ChatThread: React.FC<ChatThreadProps> = ({
  messages,
  streaming,
  disabled,
  suggestions,
  onSend,
  variant = 'inline',
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
    <div className={['flex flex-col', variant === 'popover' ? 'h-full' : ''].join(' ')}>
      {/* Message list */}
      <div
        ref={scrollRef}
        className={[
          'flex flex-col gap-3 overflow-y-auto',
          variant === 'popover' ? 'flex-1 pr-1' : 'max-h-[420px]',
        ].join(' ')}
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
                  : 'bg-bg text-text border border-border',
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

      {/* Composer */}
      <form
        className="flex items-center gap-2 mt-3"
        onSubmit={(e) => { e.preventDefault(); submit(draft); }}
      >
        <input
          type="text"
          value={draft}
          disabled={disabled}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={disabled ? 'Run inference to ask…' : 'Ask a question…'}
          aria-label="Ask about this result"
          className="flex-1 rounded-lg bg-bg border border-border px-3 py-2 text-sm font-body text-text placeholder:text-muted/60 focus:outline-none focus:border-accent/50 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={disabled || streaming || !draft.trim()}
          aria-label="Send message"
          className="w-9 h-9 rounded-lg bg-accent text-white flex items-center justify-center flex-shrink-0 disabled:bg-accent/40 disabled:cursor-not-allowed transition-colors duration-panel ease-out"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" fill="currentColor" stroke="none" />
          </svg>
        </button>
      </form>
    </div>
  );
};

export default ChatThread;
