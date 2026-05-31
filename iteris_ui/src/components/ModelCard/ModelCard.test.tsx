import { render, screen, fireEvent } from '@testing-library/react';
import { ModelCard } from './ModelCard';
import type { ModelRecord } from '@/api/contract';

const mockModel: ModelRecord = {
  id: 'ddqn',
  name: 'Double DQN',
  family: 'discrete-drl',
  description: 'Double DQN with prioritised replay for cardiac segmentation.',
  diceCamus: 0.912,
  diceBrisc: null,
  iou: 0.854,
  hd: 8.2,
  deployed: true,
  selectable: true,
};

describe('ModelCard', () => {
  it('renders model name and family badge', () => {
    render(<ModelCard model={mockModel} />);
    expect(screen.getByText('Double DQN')).toBeInTheDocument();
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
    expect(onSelect).toHaveBeenCalledWith('ddqn');
  });

  it('is not interactive when not selectable', () => {
    const onSelect = vi.fn();
    const nonSelectable: ModelRecord = { ...mockModel, selectable: false };
    render(<ModelCard model={nonSelectable} onSelect={onSelect} />);
    expect(screen.queryByRole('button')).toBeNull();
  });
});
