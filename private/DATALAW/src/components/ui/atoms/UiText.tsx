import type { ElementType, HTMLAttributes, ReactNode } from 'react';

const sizeClass: Record<'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'hero', string> = {
  xs: 'text-[length:var(--text-xs)]',
  sm: 'text-[length:var(--text-sm)]',
  md: 'text-[length:var(--text-md)]',
  lg: 'text-[length:var(--text-lg)]',
  xl: 'text-[length:var(--text-xl)]',
  hero: 'text-[length:var(--text-hero)]',
};

const toneClass: Record<'default' | 'secondary' | 'muted', string> = {
  default: 'text-[var(--app-text)]',
  secondary: 'text-[var(--app-text-secondary)]',
  muted: 'text-[var(--app-text-muted)]',
};

export type UiTextProps<T extends ElementType = 'span'> = {
  as?: T;
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'hero';
  /** Заголовочный интерлиньяж 1.2 */
  heading?: boolean;
  tone?: 'default' | 'secondary' | 'muted';
  weight?: 'normal' | 'medium' | 'semibold' | 'bold' | 'black';
  uppercase?: boolean;
  tracking?: 'normal' | 'tight' | 'widest';
  className?: string;
  children?: ReactNode;
} & Omit<HTMLAttributes<HTMLElement>, 'as' | 'children'>;

const weightClass: Record<NonNullable<UiTextProps['weight']>, string> = {
  normal: 'font-normal',
  medium: 'font-medium',
  semibold: 'font-semibold',
  bold: 'font-bold',
  black: 'font-black',
};

const trackingClass: Record<NonNullable<UiTextProps['tracking']>, string> = {
  normal: 'tracking-normal',
  tight: 'tracking-tight',
  widest: 'tracking-widest',
};

export function UiText<T extends ElementType = 'span'>({
  as,
  size = 'sm',
  heading = false,
  tone = 'default',
  weight = 'normal',
  uppercase = false,
  tracking = 'normal',
  className = '',
  children,
  ...rest
}: UiTextProps<T>) {
  const Component = (as ?? 'span') as ElementType;
  const lh = heading ? 'leading-[var(--leading-heading)]' : 'leading-[var(--leading-body)]';
  return (
    <Component
      className={[
        sizeClass[size as keyof typeof sizeClass] || sizeClass.sm,
        lh,
        toneClass[tone],
        weightClass[weight],
        trackingClass[tracking],
        uppercase ? 'uppercase' : '',
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
