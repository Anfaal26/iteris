import { render, screen, fireEvent } from '@testing-library/react';
import { IterationPlaybackTimeline } from './IterationPlaybackTimeline';
import type { IterationStep } from '@/api/contract';

const steps: IterationStep[] = [
  { step: 0, masks: [], deltaDice: 0.0, annotation: 'Initial segmentation' },
  { step: 1, masks: [], deltaDice: 0.05, annotation: 'Refining boundaries' },
  { step: 2, masks: [], deltaDice: 0.03, annotation: 'Final adjustment' },
];

describe('IterationPlaybackTimeline', () => {
  it('renders step counter', () => {
    render(<IterationPlaybackTimeline steps={steps} currentStep={0} />);
    expect(screen.getByText('1')).toBeInTheDocument();
  });

  it('renders step annotation', () => {
    render(<IterationPlaybackTimeline steps={steps} currentStep={0} />);
    expect(screen.getByText('Initial segmentation')).toBeInTheDocument();
  });

  it('calls onStepChange when next is clicked', () => {
    const onChange = vi.fn();
    render(
      <IterationPlaybackTimeline steps={steps} currentStep={0} onStepChange={onChange} />
    );
    fireEvent.click(screen.getByRole('button', { name: 'Next step' }));
    expect(onChange).toHaveBeenCalledWith(1);
  });

  it('renders play/pause button', () => {
    render(<IterationPlaybackTimeline steps={steps} currentStep={0} />);
    expect(screen.getByRole('button', { name: 'Play iteration' })).toBeInTheDocument();
  });

  it('renders speed selector buttons', () => {
    render(<IterationPlaybackTimeline steps={steps} currentStep={0} />);
    expect(screen.getByRole('button', { name: /1×/ })).toBeInTheDocument();
  });
});
