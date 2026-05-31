import { render, screen, fireEvent, act } from '@testing-library/react';
import { ToastProvider, useToast, ToastItem } from './Toast';
import type { ToastMessage } from './Toast';
import React from 'react';

const mockToast: ToastMessage = {
  id: 'test-1',
  variant: 'error',
  message: 'Something went wrong',
  timeout: 10000,
};

describe('ToastItem', () => {
  it('renders with role=alert', () => {
    render(<ToastItem toast={mockToast} onDismiss={() => {}} />);
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('calls onDismiss when dismiss button clicked', () => {
    const onDismiss = vi.fn();
    render(<ToastItem toast={mockToast} onDismiss={onDismiss} />);
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss notification' }));
    expect(onDismiss).toHaveBeenCalledWith('test-1');
  });
});

const TestConsumer: React.FC = () => {
  const { addToast } = useToast();
  return (
    <button onClick={() => addToast({ variant: 'warning', message: 'Watch out!' })}>
      Add Toast
    </button>
  );
};

describe('ToastProvider + useToast', () => {
  it('renders toasts when addToast is called', () => {
    render(
      <ToastProvider>
        <TestConsumer />
      </ToastProvider>
    );
    act(() => { fireEvent.click(screen.getByText('Add Toast')); });
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText('Watch out!')).toBeInTheDocument();
  });
});
