import { render, screen } from '@testing-library/react';
import { StructureRow } from './StructureRow';
import type { StructureMetrics } from '@/api/contract';

const mockMetrics: StructureMetrics = {
  structure: 'lv_endo',
  label: 'LV Endocardium',
  dice: 0.923,
  iou: 0.857,
  hd: 7.2,
  hd95: 5.1,
};

describe('StructureRow', () => {
  it('renders structure label', () => {
    render(<StructureRow metrics={mockMetrics} />);
    expect(screen.getByText('LV Endocardium')).toBeInTheDocument();
  });

  it('renders Dice and HD values', () => {
    render(<StructureRow metrics={mockMetrics} />);
    expect(screen.getByText('0.923')).toBeInTheDocument();
    expect(screen.getByText('7.2')).toBeInTheDocument();
  });

  it('renders without hatch swatch by default', () => {
    const { container } = render(<StructureRow metrics={mockMetrics} />);
    // No hatch pattern SVG by default
    expect(container.querySelector('pattern')).toBeNull();
  });

  it('renders hatch swatch when hatchSwatch=true', () => {
    const { container } = render(<StructureRow metrics={mockMetrics} hatchSwatch />);
    expect(container.querySelector('pattern')).toBeTruthy();
  });
});
