import { render, screen } from '@testing-library/react';
import { LogoMark } from './LogoMark';

describe('LogoMark', () => {
  it('renders with default aria-label', () => {
    render(<LogoMark />);
    expect(screen.getByRole('img', { name: 'Iteris logo' })).toBeInTheDocument();
  });

  it('renders with custom aria-label', () => {
    render(<LogoMark ariaLabel="Custom label" />);
    expect(screen.getByRole('img', { name: 'Custom label' })).toBeInTheDocument();
  });

  it('renders with custom size', () => {
    render(<LogoMark size={48} />);
    const svg = screen.getByRole('img');
    expect(svg).toHaveAttribute('width', '48');
    expect(svg).toHaveAttribute('height', '48');
  });
});
