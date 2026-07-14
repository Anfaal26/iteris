import { render, screen, fireEvent } from '@testing-library/react';
import { ModelCard } from './ModelCard';
import type { ModelRecord } from '@/api/contract';

const mockModel: ModelRecord = {
  id: 'dueling-dqn',
  name: 'Dueling DQN',
  family: 'discrete-drl',
  description: 'Dueling DQN with value/advantage streams for cardiac segmentation.',
  diceCamus: null,
  diceBrisc: null,
  iou: null,
  hd: null,
  deployed: false,
  selectable: false,
};

describe('ModelCard', () => {
  it('renders model name and family badge', () => {
    render(<ModelCard model={mockModel} />);
    expect(screen.getByText('Dueling DQN')).toBeInTheDocument();
    expect(screen.getByText('Discrete DRL')).toBeInTheDocument();
  });

  it('shows BEST badge when isBest=true', () => {
    render(<ModelCard model={mockModel} isBest />);
    expect(screen.getByText('BEST')).toBeInTheDocument();
  });

  it('calls onSelect when clicked', () => {
    const onSelect = vi.fn();
    render(<ModelCard model={mockModel} onSelect={onSelect} />);
    fireEvent.click(screen.getByRole('button'));
    expect(onSelect).toHaveBeenCalledWith('dueling-dqn');
  });

  it('is not interactive when not selectable', () => {
    const onSelect = vi.fn();
    const nonSelectable: ModelRecord = { ...mockModel, selectable: false };
    render(<ModelCard model={nonSelectable} onSelect={onSelect} />);
    expect(screen.queryByRole('button')).toBeNull();
  });
});
