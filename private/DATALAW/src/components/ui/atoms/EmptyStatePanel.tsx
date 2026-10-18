import type { ReactNode } from 'react';

export type EmptyStatePanelProps = {
  children: ReactNode;
  className?: string;
};

/**
 * Нейтральная панель подсказки / пустого состояния (Mica-подобная поверхность).
 */
export function EmptyStatePanel({ children, className = '' }: EmptyStatePanelProps) {
  return (
    <div
      className={[
        'rounded-[var(--app-radius-md)] border border-[var(--app-border)] bg-[var(--app-surface-soft)]',
        'p-6 text-center text-[length:var(--text-sm)] leading-[var(--leading-body)] text-[var(--app-text-secondary)]',
        className,
      ].join(' ')}
    >
      {children}
    </div>
  );
}
