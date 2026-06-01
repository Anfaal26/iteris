/**
 * Landing page render tests.
 *
 * Three.js (HeroScene) is lazy-loaded; we mock it out so the jsdom
 * environment doesn't have to handle WebGL.
 */

import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';

// Mock Three.js to avoid WebGL in jsdom
vi.mock('three', () => ({
  WebGLRenderer: vi.fn(() => ({
    setPixelRatio: vi.fn(),
    setClearColor: vi.fn(),
    setSize: vi.fn(),
    render: vi.fn(),
    dispose: vi.fn(),
    domElement: document.createElement('canvas'),
  })),
  Scene: vi.fn(() => ({ add: vi.fn() })),
  PerspectiveCamera: vi.fn(() => ({
    position: { set: vi.fn() },
    aspect: 1,
    updateProjectionMatrix: vi.fn(),
    rotation: { x: 0, y: 0 },
  })),
  BufferGeometry: vi.fn(() => ({
    setAttribute: vi.fn(),
    dispose: vi.fn(),
  })),
  BufferAttribute: vi.fn(),
  PointsMaterial: vi.fn(() => ({ opacity: 0.9, dispose: vi.fn() })),
  LineBasicMaterial: vi.fn(() => ({ dispose: vi.fn() })),
  Points: vi.fn(() => ({ rotation: { x: 0, y: 0 } })),
  LineSegments: vi.fn(() => ({ rotation: { x: 0, y: 0 } })),
  Color: vi.fn(),
  MathUtils: { lerp: (a: number, _b: number, _t: number) => a },
}));

// Mock the YAML import
vi.mock('@/content/models.yaml', () => ({
  default: [
    {
      id: 'ddpg',
      name: 'DDPG',
      family: 'continuous-drl',
      description: 'Test',
      diceCamus: 0.912,
      diceBrisc: 0.840,
      iou: 0.85,
      hd: 3.9,
      deployed: true,
      selectable: true,
    },
    {
      id: 'unet-baseline',
      name: 'U-Net Baseline',
      family: 'baseline',
      description: 'Test',
      diceCamus: 0.89,
      diceBrisc: 0.81,
      iou: 0.80,
      hd: 5.6,
      deployed: true,
      selectable: false,
    },
  ],
}));

// Import after mocks
import Landing from './index';

describe('Landing page', () => {
  it('renders the page without crashing', () => {
    render(<Landing />);
    // Main landmark present
    expect(screen.getByRole('main')).toBeInTheDocument();
  });

  it('renders the Hero H1 heading', () => {
    render(<Landing />);
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
    expect(screen.getByText(/See How AI/i)).toBeInTheDocument();
  });

  it('renders the eyebrow chip with capstone text', () => {
    render(<Landing />);
    // Multiple elements mention PRJ63504 — verify at least one exists
    const elements = screen.getAllByText(/PRJ63504 Capstone/i);
    expect(elements.length).toBeGreaterThan(0);
  });

  it('renders the Navbar with dark variant (banner role)', () => {
    render(<Landing />);
    expect(screen.getByRole('banner')).toBeInTheDocument();
  });

  it('renders Try Iteris CTA links to /workspace', () => {
    render(<Landing />);
    const links = screen.getAllByRole('link', { name: /Try Iteris/i });
    expect(links.length).toBeGreaterThan(0);
    expect(links[0]).toHaveAttribute('href', '/workspace');
  });

  it('renders View Our Research link to /research', () => {
    render(<Landing />);
    const link = screen.getByRole('link', { name: /View Our Research/i });
    expect(link).toHaveAttribute('href', '/research');
  });

  it('renders the stat pill with best Dice score', () => {
    render(<Landing />);
    expect(screen.getByText(/Best Dice 0\.912/i)).toBeInTheDocument();
  });

  it('renders Feature Strip section', () => {
    render(<Landing />);
    expect(screen.getByRole('region', { name: /features/i })).toBeInTheDocument();
    expect(screen.getByText('Iteration Playback')).toBeInTheDocument();
    expect(screen.getByText('Wipe Comparison')).toBeInTheDocument();
  });

  it('renders How It Works section with 4 steps', () => {
    render(<Landing />);
    expect(screen.getByRole('region', { name: /how it works/i })).toBeInTheDocument();
    expect(screen.getByText('01')).toBeInTheDocument();
    expect(screen.getByText('04')).toBeInTheDocument();
  });

  it('renders Research Metrics section', () => {
    render(<Landing />);
    expect(screen.getByRole('region', { name: /research metrics/i })).toBeInTheDocument();
    expect(screen.getByText('0.912')).toBeInTheDocument();
  });

  it('renders Model Preview with DDPG BEST badge', () => {
    render(<Landing />);
    expect(screen.getByRole('region', { name: /model preview/i })).toBeInTheDocument();
    expect(screen.getByText('BEST')).toBeInTheDocument();
  });

  it('renders Research Context pull quote', () => {
    render(<Landing />);
    expect(screen.getByRole('region', { name: /research context/i })).toBeInTheDocument();
    expect(screen.getByRole('blockquote')).toBeInTheDocument();
  });

  it('renders Pre-footer CTA section', () => {
    render(<Landing />);
    expect(screen.getByRole('region', { name: /call to action/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /Ready to explore\?/i })).toBeInTheDocument();
  });

  it('renders Footer with contentinfo role', () => {
    render(<Landing />);
    expect(screen.getByRole('contentinfo')).toBeInTheDocument();
  });

  it('renders footer columns', () => {
    render(<Landing />);
    expect(screen.getByText('Product')).toBeInTheDocument();
    // "Research" appears in both nav and footer heading — use getAllByText
    expect(screen.getAllByText('Research').length).toBeGreaterThan(0);
    expect(screen.getByText('Resources')).toBeInTheDocument();
    expect(screen.getByText('Project')).toBeInTheDocument();
  });
});
