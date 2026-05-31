import { render, screen, fireEvent } from '@testing-library/react';
import { ReadingRoomToggle } from './ReadingRoomToggle';

describe('ReadingRoomToggle', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
  });

  it('renders with initial clinical theme label', () => {
    render(<ReadingRoomToggle />);
    expect(screen.getByRole('button', { name: /reading-room/i })).toBeInTheDocument();
  });

  it('toggles to reading-room theme when clicked', () => {
    render(<ReadingRoomToggle />);
    const btn = screen.getByRole('button');
    fireEvent.click(btn);
    expect(document.documentElement.getAttribute('data-theme')).toBe('reading-room');
  });

  it('has correct aria-pressed state', () => {
    render(<ReadingRoomToggle />);
    const btn = screen.getByRole('button');
    expect(btn).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(btn);
    expect(btn).toHaveAttribute('aria-pressed', 'true');
  });
});
