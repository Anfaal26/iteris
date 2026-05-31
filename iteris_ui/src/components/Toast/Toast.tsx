/**
 * Toast — error/warning notification with auto-dismiss and dismiss button.
 * Includes ToastProvider and useToast hook for imperative usage.
 */

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useRef,
  useEffect,
  type ReactNode,
} from 'react';

/** Severity level of a toast. */
export type ToastVariant = 'error' | 'warning';

/** A single toast message. */
export interface ToastMessage {
  id: string;
  variant: ToastVariant;
  message: string;
  /** Auto-dismiss after this many ms. @default 5000 */
  timeout?: number;
}

interface ToastContextValue {
  addToast: (msg: Omit<ToastMessage, 'id'>) => void;
  removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

let _idCounter = 0;

/** Individual toast item. */
export interface ToastItemProps {
  toast: ToastMessage;
  onDismiss: (id: string) => void;
}

const VARIANT_STYLES: Record<ToastVariant, string> = {
  error: 'border-error bg-error/10 text-error',
  warning: 'border-warning bg-warning/10 text-warning',
};

const VARIANT_ICON: Record<ToastVariant, ReactNode> = {
  error: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 3.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 8 4.5zm0 7a.75.75 0 1 1 0-1.5.75.75 0 0 1 0 1.5z" />
    </svg>
  ),
  warning: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.71c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z" />
    </svg>
  ),
};

/** Single rendered toast. */
export const ToastItem: React.FC<ToastItemProps> = ({ toast, onDismiss }) => {
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const ms = toast.timeout ?? 5000;
    timeoutRef.current = setTimeout(() => onDismiss(toast.id), ms);
    return () => { if (timeoutRef.current) clearTimeout(timeoutRef.current); };
  }, [toast.id, toast.timeout, onDismiss]);

  return (
    <div
      role="alert"
      aria-live="assertive"
      className={[
        'flex items-start gap-2 px-3 py-2.5 rounded-lg border shadow-lg max-w-sm w-full',
        'transition-all duration-slide ease-out',
        VARIANT_STYLES[toast.variant],
      ].join(' ')}
    >
      <span className="flex-shrink-0 mt-0.5">{VARIANT_ICON[toast.variant]}</span>
      <span className="flex-1 text-sm font-body">{toast.message}</span>
      <button
        type="button"
        onClick={() => onDismiss(toast.id)}
        aria-label="Dismiss notification"
        className="flex-shrink-0 opacity-70 hover:opacity-100 transition-opacity duration-panel ease-out"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
          <line x1="1" y1="1" x2="13" y2="13" />
          <line x1="13" y1="1" x2="1" y2="13" />
        </svg>
      </button>
    </div>
  );
};

/** Props for ToastProvider. */
export interface ToastProviderProps {
  children: ReactNode;
}

/**
 * Provider that renders the toast stack and exposes `useToast()`.
 * Place near the root of your app.
 */
export const ToastProvider: React.FC<ToastProviderProps> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const addToast = useCallback((msg: Omit<ToastMessage, 'id'>) => {
    const id = `toast-${++_idCounter}`;
    setToasts((prev) => [...prev, { ...msg, id }]);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ addToast, removeToast }}>
      {children}
      {/* Toast stack — fixed bottom-right */}
      <div
        aria-label="Notifications"
        className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none"
      >
        {toasts.map((t) => (
          <div key={t.id} className="pointer-events-auto">
            <ToastItem toast={t} onDismiss={removeToast} />
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
};

/**
 * Hook to imperatively add toasts from any component inside `ToastProvider`.
 * @throws if used outside `ToastProvider`.
 */
export function useToast(): Pick<ToastContextValue, 'addToast'> {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return { addToast: ctx.addToast };
}

export default ToastProvider;
