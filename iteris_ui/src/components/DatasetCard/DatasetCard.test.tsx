import { render, screen, fireEvent } from '@testing-library/react';
import { DatasetCard } from './DatasetCard';

describe('DatasetCard', () => {
  const props = {
    datasetId: 'camus' as const,
    name: 'CAMUS',
    modality: 'ultrasound' as const,
    description: 'Cardiac acquisition from multiple sites.',
    sampleCount: 500,
    bestDice: 0.917,
  };

  it('renders name and modality', () => {
    render(<DatasetCard {...props} />);
    expect(screen.getByText('CAMUS')).toBeInTheDocument();
    expect(screen.getByText('ultrasound')).toBeInTheDocument();
  });

  it('shows best Dice badge', () => {
    render(<DatasetCard {...props} />);
    expect(screen.getByText('Best Dice 0.917')).toBeInTheDocument();
  });

  it('calls onSelect when clicked', () => {
    const onSelect = vi.fn();
    render(<DatasetCard {...props} onSelect={onSelect} />);
    fireEvent.click(screen.getByRole('button'));
    expect(onSelect).toHaveBeenCalledWith('camus');
  });
});
