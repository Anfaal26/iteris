import { render, screen } from '@testing-library/react';
import { PreprocessingStepIndicator } from './PreprocessingStepIndicator';

const steps = [
  { label: 'Load', done: true, elapsedMs: 12 },
  { label: 'Normalise', done: true, elapsedMs: 45 },
  { label: 'Resize', done: false },
  { label: 'Augment', done: false },
  { label: 'Ready', done: false },
];

describe('PreprocessingStepIndicator', () => {
  it('renders all 5 step labels', () => {
    render(<PreprocessingStepIndicator steps={steps} />);
    expect(screen.getByText('Load')).toBeInTheDocument();
    expect(screen.getByText('Normalise')).toBeInTheDocument();
    expect(screen.getByText('Resize')).toBeInTheDocument();
  });

  it('shows elapsed ms for completed steps', () => {
    render(<PreprocessingStepIndicator steps={steps} />);
    expect(screen.getByText('12ms')).toBeInTheDocument();
    expect(screen.getByText('45ms')).toBeInTheDocument();
  });

  it('has accessible list role', () => {
    render(<PreprocessingStepIndicator steps={steps} />);
    expect(screen.getByRole('list', { name: 'Preprocessing pipeline' })).toBeInTheDocument();
  });
});
