import { render, screen } from '@testing-library/react';
import { Navbar } from './Navbar';

describe('Navbar', () => {
  it('renders logo link', () => {
    render(<Navbar />);
    expect(screen.getByRole('link', { name: 'Iteris home' })).toBeInTheDocument();
  });

  it('renders nav items', () => {
    render(<Navbar navItems={[{ label: 'Research', href: '/research' }]} />);
    expect(screen.getByRole('link', { name: 'Research' })).toBeInTheDocument();
  });

  it('renders search button when onSearch provided', () => {
    render(<Navbar onSearch={() => {}} />);
    expect(screen.getByRole('button', { name: 'Open search' })).toBeInTheDocument();
  });

  it('renders settings button when onSettings provided', () => {
    render(<Navbar onSettings={() => {}} />);
    expect(screen.getByRole('button', { name: 'Open settings' })).toBeInTheDocument();
  });

  it('has banner role', () => {
    render(<Navbar />);
    expect(screen.getByRole('banner')).toBeInTheDocument();
  });
});
