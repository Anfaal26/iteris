import { render, screen } from '@testing-library/react';
import { MetricCard } from './MetricCard';

describe('MetricCard', () => {
  it('renders label and value', () => {
    render(
      <MetricCard label="Dice" value="0.923" status="success" baselineLabel="Baseline: 0.841" />
    );
    expect(screen.getByText('Dice')).toBeInTheDocument();
    expect(screen.getByText('0.923')).toBeInTheDocument();
  });

  it('shows baseline label', () => {
    render(
      <MetricCard label="IoU" value="0.867" status="warning" baselineLabel="Baseline: 0.870" />
    );
    expect(screen.getByText('Baseline: 0.870')).toBeInTheDocument();
  });

  it('renders status dot with aria-label', () => {
    render(
      <MetricCard label="HD" value="12.3" status="error" baselineLabel="Baseline: 10.1" />
    );
    expect(screen.getByTitle('Below baseline')).toBeInTheDocument();
  });
});
