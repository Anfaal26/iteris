/**
 * Navbar — frosted-glass top navigation bar (redesigned for medical-imaging aesthetic).
 *
 * Dark variant (landing): always frosted ice-blue glass. A "mixed boundary" gradient
 * div bleeds 56px below the bar, dissolving the edge softly into the page.
 * Scroll past 80px ramps opacity from 52% → 82%.
 *
 * Light variant (workspace / research): clean clinical surface bar.
 */

import React, { useState, useEffect } from 'react';
import { LogoMark } from '@/components/LogoMark/LogoMark';
import { ReadingRoomToggle } from '@/components/ReadingRoomToggle/ReadingRoomToggle';

export interface NavItem {
  label: string;
  href: string;
}

export interface NavbarProps {
  variant?: 'dark' | 'light';
  navItems?: NavItem[];
  onSearch?: () => void;
  onSettings?: () => void;
  className?: string;
}

const SCROLL_THRESHOLD = 80;

export const Navbar: React.FC<NavbarProps> = ({
  variant = 'dark',
  navItems = [],
  onSearch,
  onSettings,
  className,
}) => {
  const [scrolled, setScrolled] = useState(false);
  const isDark = variant === 'dark';

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > SCROLL_THRESHOLD);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const headerStyle: React.CSSProperties = isDark
    ? {
        height: '64px',
        background: scrolled ? 'rgba(3, 5, 8, 0.82)' : 'rgba(3, 5, 8, 0.52)',
        backdropFilter: 'blur(24px) saturate(180%)',
        WebkitBackdropFilter: 'blur(24px) saturate(180%)',
        borderBottom: scrolled
          ? '1px solid rgba(186, 230, 253, 0.08)'
          : '1px solid transparent',
      }
    : {
        height: '56px',
        background: 'var(--color-surface)',
        borderBottom: '1px solid var(--color-border)',
      };

  const linkCls = isDark
    ? 'text-[13px] font-body text-landing-text/60 hover:text-landing-text transition-colors duration-panel ease-out no-underline'
    : 'text-[13px] font-body text-muted hover:text-text transition-colors duration-panel ease-out no-underline';

  const iconCls = [
    'flex items-center justify-center w-8 h-8 rounded-full transition-colors duration-panel ease-out',
    isDark
      ? 'text-landing-text/50 hover:text-landing-text hover:bg-white/[0.06]'
      : 'text-muted hover:text-text hover:bg-black/[0.04]',
  ].join(' ');

  return (
    <>
      <header
        role="banner"
        className={[
          'fixed top-0 inset-x-0 z-50 flex items-center justify-between px-6 lg:px-10',
          'transition-all duration-slide ease-out',
          isDark ? 'text-landing-text' : 'text-text',
          className ?? '',
        ].join(' ')}
        style={headerStyle}
      >
        {/* Left — logo + wordmark */}
        <a href="/" className="flex items-center gap-2.5 no-underline flex-shrink-0" aria-label="Iteris home">
          <LogoMark size={26} className={isDark ? 'text-grad-b' : 'text-accent'} ariaLabel="" />
          <span className={[
            'font-heading font-semibold tracking-[0.06em] text-sm',
            isDark ? 'text-landing-text' : 'text-text',
          ].join(' ')}>
            ITERIS
          </span>
        </a>

        {/* Centre — nav links */}
        {navItems.length > 0 && (
          <nav aria-label="Main navigation" className="hidden md:block">
            <ul className="flex items-center gap-8 list-none m-0 p-0">
              {navItems.map((item) => (
                <li key={item.href}>
                  <a href={item.href} className={linkCls}>{item.label}</a>
                </li>
              ))}
            </ul>
          </nav>
        )}

        {/* Right — icons + CTA */}
        <div className="flex items-center gap-1 flex-shrink-0">
          <ReadingRoomToggle className={isDark ? 'text-landing-text/50 hover:text-landing-text' : ''} />

          {onSearch && (
            <button type="button" onClick={onSearch} aria-label="Open search" className={iconCls}>
              <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
                <circle cx="6.5" cy="6.5" r="4.5" /><line x1="10" y1="10" x2="14" y2="14" />
              </svg>
            </button>
          )}

          {onSettings && (
            <button type="button" onClick={onSettings} aria-label="Open settings" className={iconCls}>
              <svg width="15" height="15" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
              </svg>
            </button>
          )}

          {/* Primary CTA pill — dark variant only */}
          {isDark && (
            <a
              href="/workspace"
              className="ml-3 hidden sm:flex items-center gap-1.5 rounded-full px-4 py-1.5
                         text-[12px] font-heading font-semibold tracking-wide
                         border text-grad-b
                         hover:bg-grad-b/10
                         transition-all duration-panel ease-out no-underline"
              style={{ borderColor: 'rgba(56,189,248,0.3)', background: 'rgba(56,189,248,0.07)' }}
            >
              Try Iteris
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
                <path d="M2 5h6M5 2l3 3-3 3" />
              </svg>
            </a>
          )}
        </div>
      </header>

      {/* Mixed boundary — hazy gradient that bleeds from nav edge into page (dark only).
          This dissolves the nav bottom edge rather than leaving a hard line. */}
      {isDark && (
        <div
          aria-hidden="true"
          className="fixed inset-x-0 z-40 pointer-events-none"
          style={{
            top: '64px',
            height: '60px',
            background: 'linear-gradient(to bottom, rgba(3,5,8,0.4) 0%, transparent 100%)',
          }}
        />
      )}
    </>
  );
};

export default Navbar;
