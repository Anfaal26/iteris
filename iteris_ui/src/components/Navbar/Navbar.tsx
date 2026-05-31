/**
 * Navbar — top navigation bar with dark and light variants.
 * Transitions from transparent to frosted glass after 80 px of scroll.
 * Logo + wordmark on the left, centre links, right icon pills.
 */

import React, { useState, useEffect } from 'react';
import { LogoMark } from '@/components/LogoMark/LogoMark';
import { ReadingRoomToggle } from '@/components/ReadingRoomToggle/ReadingRoomToggle';

/** A single nav link item. */
export interface NavItem {
  label: string;
  href: string;
}

/** Props for the Navbar component. */
export interface NavbarProps {
  /** Visual variant. @default 'dark' */
  variant?: 'dark' | 'light';
  /** Centre navigation links. */
  navItems?: NavItem[];
  /** Called when the search icon is activated. */
  onSearch?: () => void;
  /** Called when the settings icon is activated. */
  onSettings?: () => void;
  /** Additional class names. */
  className?: string;
}

const SCROLL_THRESHOLD = 80;

/**
 * Primary navigation bar for the ITERIS workstation.
 * Becomes frosted glass (backdrop-blur + border-bottom) after 80 px scroll.
 */
export const Navbar: React.FC<NavbarProps> = ({
  variant = 'dark',
  navItems = [],
  onSearch,
  onSettings,
  className,
}) => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > SCROLL_THRESHOLD);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const isDark = variant === 'dark';

  const frostedStyle: React.CSSProperties = scrolled
    ? {
        backgroundColor: 'rgba(5,7,12,0.85)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        borderBottom: '1px solid var(--border)',
      }
    : {};

  return (
    <header
      role="banner"
      className={[
        'fixed top-0 inset-x-0 z-50 flex items-center justify-between px-6',
        'h-navbar transition-all duration-panel ease-out',
        isDark ? 'text-landing-text' : 'text-text',
        !scrolled && isDark ? 'bg-transparent' : '',
        !scrolled && !isDark ? 'bg-surface border-b border-border' : '',
        className ?? '',
      ].join(' ')}
      style={scrolled ? frostedStyle : undefined}
    >
      {/* Left: logo + wordmark */}
      <a
        href="/"
        className="flex items-center gap-2 font-heading font-semibold text-base no-underline"
        aria-label="Iteris home"
      >
        <LogoMark
          size={28}
          className={isDark ? 'text-grad-a' : 'text-accent'}
          ariaLabel=""
        />
        <span className={isDark ? 'text-landing-text' : 'text-text'}>ITERIS</span>
      </a>

      {/* Centre: nav links */}
      {navItems.length > 0 && (
        <nav aria-label="Main navigation">
          <ul className="flex items-center gap-6 list-none m-0 p-0">
            {navItems.map((item) => (
              <li key={item.href}>
                <a
                  href={item.href}
                  className={[
                    'text-sm font-body no-underline transition-colors duration-panel ease-out',
                    isDark
                      ? 'text-landing-text/70 hover:text-landing-text'
                      : 'text-muted hover:text-text',
                  ].join(' ')}
                >
                  {item.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      )}

      {/* Right: icon pills */}
      <div className="flex items-center gap-1">
        <ReadingRoomToggle className={isDark ? 'text-landing-text/70 hover:text-landing-text' : ''} />

        {onSearch && (
          <button
            type="button"
            onClick={onSearch}
            aria-label="Open search"
            className={[
              'flex items-center justify-center w-8 h-8 rounded-md',
              'transition-colors duration-panel ease-out',
              isDark
                ? 'text-landing-text/70 hover:text-landing-text'
                : 'text-muted hover:text-text',
            ].join(' ')}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
              <circle cx="6.5" cy="6.5" r="4.5" />
              <line x1="10" y1="10" x2="14" y2="14" />
            </svg>
          </button>
        )}

        {onSettings && (
          <button
            type="button"
            onClick={onSettings}
            aria-label="Open settings"
            className={[
              'flex items-center justify-center w-8 h-8 rounded-md',
              'transition-colors duration-panel ease-out',
              isDark
                ? 'text-landing-text/70 hover:text-landing-text'
                : 'text-muted hover:text-text',
            ].join(' ')}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
              <path d="M8 10a2 2 0 1 0 0-4 2 2 0 0 0 0 4zm6.53-2.65a6.1 6.1 0 0 0 .05-.35l1-.78a.25.25 0 0 0 .06-.32l-.94-1.63a.25.25 0 0 0-.3-.11l-1.17.47a5.5 5.5 0 0 0-.6-.35l-.18-1.24A.25.25 0 0 0 12.2 3H10.8a.25.25 0 0 0-.25.21l-.18 1.24c-.2.1-.4.22-.6.35L8.6 4.33a.25.25 0 0 0-.3.11L7.36 6.07a.25.25 0 0 0 .06.32l1 .78c-.02.12-.04.23-.04.35s.02.23.04.35l-1 .78a.25.25 0 0 0-.06.32l.94 1.63c.06.11.19.15.3.11l1.17-.47c.19.13.39.25.6.35l.18 1.24c.04.13.15.21.25.21h1.4c.11 0 .21-.08.25-.21l.18-1.24c.2-.1.4-.22.6-.35l1.17.47c.11.04.24 0 .3-.11l.94-1.63a.25.25 0 0 0-.06-.32l-1-.78z" />
            </svg>
          </button>
        )}
      </div>
    </header>
  );
};

export default Navbar;
