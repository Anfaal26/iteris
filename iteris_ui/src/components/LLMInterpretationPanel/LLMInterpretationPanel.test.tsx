import { render, screen } from '@testing-library/react';
import { LLMInterpretationPanel } from './LLMInterpretationPanel';

describe('LLMInterpretationPanel', () => {
  it('renders all 5 section headings', () => {
    render(<LLMInterpretationPanel />);
    expect(screen.getByRole('region', { name: 'Segmentation Summary' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Clinical Significance' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Metric Interpretation' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Performance Analysis' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Literature References' })).toBeInTheDocument();
  });

  it('renders disclaimer text', () => {
    render(<LLMInterpretationPanel />);
    expect(
      screen.getByText(/AI-generated interpretation for research use only/i)
    ).toBeInTheDocument();
  });

  it('renders static section content', () => {
    render(
      <LLMInterpretationPanel
        sections={{ 'segmentation-summary': 'Good segmentation quality.' }}
      />
    );
    expect(screen.getByText('Good segmentation quality.')).toBeInTheDocument();
  });

  it('renders copy button', () => {
    render(<LLMInterpretationPanel />);
    expect(screen.getByRole('button', { name: 'Copy interpretation to clipboard' })).toBeInTheDocument();
  });

  it('renders loading indicator when loading=true with no content', () => {
    render(<LLMInterpretationPanel loading />);
    expect(screen.getByText('Generating interpretation…')).toBeInTheDocument();
  });
});
