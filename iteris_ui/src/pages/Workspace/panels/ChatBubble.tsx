/**
 * ChatBubble — the persistent chat affordance fixed to the viewport corner
 * (redesign). Independent of scroll position, it opens the SAME thread the
 * inline chat section uses, for users who want to ask something immediately
 * without scrolling past the stats.
 */

import React, { useState } from 'react';
import type { ChatMessage } from '@/api/contract';
import { ChatThread } from './ChatThread';

export interface ChatBubbleProps {
  messages: ChatMessage[];
  streaming: boolean;
  disabled: boolean;
  suggestions: string[];
  onSend: (text: string) => void;
  /** Failure from the last turn, surfaced inline with a retry. */
  error?: string | null;
  onRetry?: () => void;
}

export const ChatBubble: React.FC<ChatBubbleProps> = ({
  messages,
  streaming,
  disabled,
  suggestions,
  onSend,
  error,
  onRetry,
}) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="fixed bottom-5 right-5 z-40 flex flex-col items-end gap-3">
      {open && (
        <div
          role="dialog"
          aria-label="Ask about this result"
          className="w-[340px] max-w-[calc(100vw-2.5rem)] h-[440px] max-h-[70vh] rounded-2xl bg-surface border border-border shadow-float flex flex-col overflow-hidden"
        >
          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <span className="text-sm font-heading font-semibold text-text">Ask about this result</span>
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close chat"
              className="text-muted hover:text-text transition-colors duration-panel ease-out"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <div className="flex-1 min-h-0 p-4">
            <ChatThread
              messages={messages}
              streaming={streaming}
              disabled={disabled}
              suggestions={suggestions}
              onSend={onSend}
              variant="popover"
              error={error}
              onRetry={onRetry}
            />
          </div>
        </div>
      )}

      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? 'Hide chat' : 'Ask about this result'}
        aria-expanded={open}
        className="w-12 h-12 rounded-full bg-accent text-white flex items-center justify-center shadow-lg hover:bg-accent/90 transition-colors duration-panel ease-out"
      >
        {open ? (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        ) : (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.5 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8A8.5 8.5 0 0 1 12.5 3a8.38 8.38 0 0 1 8.5 8.5z" />
          </svg>
        )}
      </button>
    </div>
  );
};

export default ChatBubble;
