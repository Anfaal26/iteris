import { render, screen, fireEvent } from '@testing-library/react';
import { SampleImageTile } from './SampleImageTile';
import type { SampleImage } from '@/api/contract';

const mockImage: SampleImage = {
  id: 'img-001',
  thumbnailB64: '',
  modality: 'ultrasound',
  anatomy: 'cardiac',
  difficulty: 'easy',
  bestDice: 0.91,
  dataset: 'camus',
};

describe('SampleImageTile', () => {
  it('renders placeholder SVG when no thumbnail', () => {
    render(<SampleImageTile image={mockImage} />);
    expect(screen.getByRole('img', { name: /placeholder/i })).toBeInTheDocument();
  });

  it('renders real thumbnail when provided', () => {
    const img = { ...mockImage, thumbnailB64: 'abc123' };
    render(<SampleImageTile image={img} />);
    expect(screen.getByRole('img', { name: /cardiac/i })).toBeInTheDocument();
  });

  it('calls onSelect when clicked', () => {
    const onSelect = vi.fn();
    render(<SampleImageTile image={mockImage} onSelect={onSelect} />);
    fireEvent.click(screen.getByRole('button'));
    expect(onSelect).toHaveBeenCalledWith('img-001');
  });

  it('has aria-pressed when selected', () => {
    render(<SampleImageTile image={mockImage} selected />);
    expect(screen.getByRole('button')).toHaveAttribute('aria-pressed', 'true');
  });
});
