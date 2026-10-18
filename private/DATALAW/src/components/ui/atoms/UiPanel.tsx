import type { ReactNode } from 'react';

export type UiPanelElevation = 'none' | 'sm' | 'md';

export type UiPanelProps = {
  children: ReactNode;
  className?: string;
  /** Mica-подобная панель: полупрозрачный фон + размытие */
  mica?: boolean;
  elevation?: UiPanelElevation;
  padding?: 'none' | 'sm' | 'md' | 'lg';
};

const elevationClass: Record<UiPanelElevation, string> = {
  none: 'shadow-none',
  sm: 'shadow-[var(--app-shadow-sm)]',
  md: 'shadow-[var(--app-shadow-md)]',
};

const paddingClass: Record<NonNullable<UiPanelProps['padding']>, string> = {
  none: '',
  sm: 'p-2',
  md: 'p-4',
  lg: 'p-6',
};

export function UiPanel({
  children,
  className = '',
  mica = true,
  elevation = 'md',
  padding = 'md',
}: UiPanelProps) {
  return (
    <div
      className={[
        'rounded-[var(--app-radius-md)]',
        mica ? 'mica-effect' : 'border border-[var(--app-border)] bg-[var(--app-surface-raised)]',
        elevationClass[elevation],
        paddingClass[padding],
        className,
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {children}
    </div>
  );
}
