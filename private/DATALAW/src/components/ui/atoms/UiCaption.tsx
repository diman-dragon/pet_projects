import type { ElementType, HTMLAttributes, ReactNode } from 'react';

export type UiCaptionProps<T extends ElementType = 'span'> = {
  as?: T;
  /** Верхний мелкий колонтитул (uppercase, плотный трекинг) */
  overline?: boolean;
  tone?: 'muted' | 'secondary';
  className?: string;
  children?: ReactNode;
} & Omit<HTMLAttributes<HTMLElement>, 'as' | 'children'>;

const toneClass: Record<NonNullable<UiCaptionProps['tone']>, string> = {
  muted: 'text-[var(--app-text-muted)]',
  secondary: 'text-[var(--app-text-secondary)]',
};

export function UiCaption<T extends ElementType = 'span'>({
  as,
  overline = false,
  tone = 'muted',
  className = '',
  children,
  ...rest
}: UiCaptionProps<T>) {
  const Component = (as ?? 'span') as ElementType;
  return (
    <Component
      className={[
        'text-[length:var(--text-xs)] leading-[var(--leading-body)]',
        overline ? 'font-black uppercase tracking-[0.18em]' : 'font-medium',
        toneClass[tone],
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      {...rest}
    >
      {children}
    </Component>
  );
}
