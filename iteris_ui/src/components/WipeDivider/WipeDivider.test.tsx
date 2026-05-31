import { render, screen, fireEvent } from '@testing-library/react';
import { WipeDivider } from './WipeDivider';

describe('WipeDivider', () => {
  it('renders with correct aria-valuenow', () => {
    render(<WipeDivider value={50} onChange={() => {}} />);
    expect(screen.getByRole('slider')).toHaveAttribute('aria-valuenow', '50');
  });

  it('double-click calls onChange(50)', () => {
    const onChange = vi.fn();
    render(<WipeDivider value={30} onChange={onChange} />);
    fireEvent.dblClick(screen.getByRole('slider'));
    expect(onChange).toHaveBeenCalledWith(50);
  });

  it('arrow keys change value', () => {
    const onChange = vi.fn();
    render(<WipeDivider value={50} onChange={onChange} />);
    const slider = screen.getByRole('slider');
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(onChange).toHaveBeenCalledWith(60);
    fireEvent.keyDown(slider, { key: 'ArrowLeft' });
    expect(onChange).toHaveBeenCalledWith(40);
  });

  it('Home/End keys snap to extremes', () => {
    const onChange = vi.fn();
    render(<WipeDivider value={50} onChange={onChange} />);
    const slider = screen.getByRole('slider');
    fireEvent.keyDown(slider, { key: 'Home' });
    expect(onChange).toHaveBeenCalledWith(0);
    fireEvent.keyDown(slider, { key: 'End' });
    expect(onChange).toHaveBeenCalledWith(100);
  });

  it('shows sync indicator when synced=true', () => {
    render(<WipeDivider value={50} synced />);
    expect(screen.getByText('Sync')).toBeInTheDocument();
  });
});
