import { render, screen, fireEvent } from '@testing-library/react';
import { ExportButtonGroup } from './ExportButtonGroup';

describe('ExportButtonGroup', () => {
  it('renders all three action buttons', () => {
    render(<ExportButtonGroup />);
    expect(screen.getByRole('button', { name: 'Download PNG mask' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Export JSON metrics' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy session link' })).toBeInTheDocument();
  });

  it('calls onDownloadPng when clicked', () => {
    const onDownload = vi.fn();
    render(<ExportButtonGroup onDownloadPng={onDownload} />);
    fireEvent.click(screen.getByRole('button', { name: 'Download PNG mask' }));
    expect(onDownload).toHaveBeenCalled();
  });

  it('calls onExportJson when clicked', () => {
    const onExport = vi.fn();
    render(<ExportButtonGroup onExportJson={onExport} />);
    fireEvent.click(screen.getByRole('button', { name: 'Export JSON metrics' }));
    expect(onExport).toHaveBeenCalled();
  });

  it('has export actions group role', () => {
    render(<ExportButtonGroup />);
    expect(screen.getByRole('group', { name: 'Export actions' })).toBeInTheDocument();
  });
});
