import { useEffect } from 'react';

export function useKeyboardShortcuts(shortcuts: {
  onEnter?: () => void;
  onEscape?: () => void;
}) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Enter' && shortcuts.onEnter) {
        event.preventDefault();
        shortcuts.onEnter();
      }
      if (event.key === 'Escape' && shortcuts.onEscape) {
        event.preventDefault();
        shortcuts.onEscape();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
}
