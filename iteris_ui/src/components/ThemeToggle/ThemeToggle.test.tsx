import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeToggle } from './ThemeToggle';

describe('ThemeToggle', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
  });

  it('renders with initial dark theme label', () => {
    render(<ThemeToggle />);
    expect(screen.getByRole('button', { name: /switch to light theme/i })).toBeInTheDocument();
  });

  it('toggles to light theme when clicked', () => {
    render(<ThemeToggle />);
    const btn = screen.getByRole('button');
    fireEvent.click(btn);
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('has correct aria-pressed state', () => {
    render(<ThemeToggle />);
    const btn = screen.getByRole('button');
    expect(btn).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(btn);
    expect(btn).toHaveAttribute('aria-pressed', 'true');
  });
});
