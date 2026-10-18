import type { ReactNode } from 'react';
import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { UiText } from '../atoms/UiText';

type AppModalSize = 'sm' | 'md' | 'lg' | 'xl';

interface AppModalProps {
  open: boolean;
  title?: ReactNode;
  onClose: () => void;
  footer?: ReactNode;
  children: ReactNode;
  size?: AppModalSize;
  showCloseButton?: boolean;
}

const sizes: Record<AppModalSize, string> = {
  sm: 'max-w-[400px]',
  md: 'max-w-[520px]',
  lg: 'max-w-[680px]',
  xl: 'max-w-[860px]',
};

export function AppModal({
  open,
  title,
  onClose,
  footer,
  children,
  size = 'md',
  showCloseButton = true,
}: AppModalProps) {
  useEffect(() => {
    document.body.style.overflow = open ? 'hidden' : 'unset';
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose, open]);

  if (!open) return null;

  return createPortal(
    <div className="fixed inset-0 z-[1000] flex items-center justify-center p-4">
      <button
        type="button"
        aria-label="Close modal"
        className="absolute inset-0 bg-[var(--app-modal-scrim)] backdrop-blur-[8px]"
        onClick={showCloseButton ? onClose : undefined}
      />
      <div
        className={[
          'relative z-10 flex w-full flex-col overflow-hidden rounded-[var(--app-radius-lg)] border border-[var(--app-border)]',
          'bg-[var(--app-panel-bg)] shadow-[var(--app-shadow-md)] backdrop-blur-xl',
          sizes[size],
        ].join(' ')}
      >
        {title ? (
          <div className="flex min-h-[var(--app-modal-header-min-h)] items-center justify-between border-b border-[var(--app-border)] px-[var(--app-modal-padding-x)] py-[var(--app-modal-padding-y-header)]">
            <div className="min-w-0">{title}</div>
            {showCloseButton ? (
              <button
                type="button"
                onClick={onClose}
                className="ml-3 flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-[var(--app-text-muted)] transition-colors hover:bg-[color-mix(in_srgb,var(--app-text)_6%,transparent)] hover:text-[var(--app-text)]"
              >
                <X size={18} />
              </button>
            ) : null}
          </div>
        ) : null}
        <div className="max-h-[75vh] overflow-y-auto px-[var(--app-modal-padding-x)] py-[var(--app-modal-padding-y-body)] custom-scroll">
          {children}
        </div>
        {footer ? (
          <div className="border-t border-[var(--app-border)] bg-[var(--app-surface-soft)] px-[var(--app-modal-padding-x)] py-[var(--app-modal-padding-y-footer)]">
            {footer}
          </div>
        ) : null}
      </div>
    </div>,
    document.body,
  );
}

export function AppModalHeader({
  title,
  subtitle,
}: {
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="min-w-0">
      <UiText as="div" size="md" weight="semibold" heading>
        {title}
      </UiText>
      {subtitle ? (
        <UiText as="div" size="xs" tone="secondary" className="mt-1">
          {subtitle}
        </UiText>
      ) : null}
    </div>
  );
}
