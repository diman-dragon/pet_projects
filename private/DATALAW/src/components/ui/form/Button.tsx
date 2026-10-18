import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { Loader2 } from 'lucide-react';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline';
export type ButtonSize = 'xs' | 'sm' | 'md' | 'lg';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: ReactNode;
  iconRight?: ReactNode;
  loading?: boolean;
}

const variantClass: Record<ButtonVariant, string> = {
  primary: [
    'border-transparent bg-[var(--app-accent)] text-[var(--app-surface)]',
    'hover:bg-[var(--app-accent-strong)]',
    'shadow-[var(--app-shadow-sm)]',
  ].join(' '),
  secondary: [
    'border-[var(--app-border)] bg-[var(--app-surface)] text-[var(--app-text)]',
    'hover:bg-[var(--app-surface-soft)] hover:border-[var(--app-border-strong)]',
  ].join(' '),
  ghost: [
    'border-transparent bg-transparent text-[var(--app-text-secondary)]',
    'hover:bg-[color-mix(in_srgb,var(--app-text)_5%,transparent)] hover:text-[var(--app-text)]',
  ].join(' '),
  danger: [
    'border-transparent bg-[var(--app-danger)] text-[var(--app-surface)]',
    'hover:bg-[color-mix(in_srgb,var(--app-danger)_88%,black)]',
    'shadow-[var(--app-shadow-sm)]',
  ].join(' '),
  outline: [
    'border-[var(--app-border)] bg-transparent text-[var(--app-text)]',
    'hover:border-[var(--app-accent)] hover:bg-[color-mix(in_srgb,var(--app-accent)_5%,transparent)]',
  ].join(' '),
};

const sizeClass: Record<ButtonSize, string> = {
  xs: 'h-7 min-w-7 gap-1.5 rounded-[var(--app-radius-sm)] px-2.5 text-[length:var(--text-xs)]',
  sm: 'h-8 min-w-8 gap-2 rounded-[var(--app-radius-sm)] px-3 text-[length:var(--text-xs)]',
  md: 'h-9 min-w-9 gap-2 rounded-[var(--app-radius-md)] px-4 text-[length:var(--text-sm)]',
  lg: 'h-11 min-w-11 gap-2.5 rounded-[var(--app-radius-md)] px-5 text-[length:var(--text-md)]',
};

export function Button({
  variant = 'secondary',
  size = 'md',
  icon,
  iconRight,
  loading = false,
  disabled = false,
  className = '',
  children,
  type = 'button',
  ...props
}: ButtonProps) {
  const iconNode = loading ? <Loader2 size={16} className="animate-spin" /> : icon;

  return (
    <button
      type={type}
      disabled={disabled || loading}
      className={[
        'inline-flex items-center justify-center border font-medium leading-[var(--leading-body)] transition-all duration-150',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color-mix(in_srgb,var(--app-accent)_28%,transparent)]',
        'disabled:pointer-events-none disabled:opacity-50',
        'active:translate-y-px',
        variantClass[variant],
        sizeClass[size],
        className,
      ].join(' ')}
      {...props}
    >
      {iconNode ? <span className="shrink-0">{iconNode}</span> : null}
      {children ? <span className="truncate">{children}</span> : null}
      {!loading && iconRight ? <span className="shrink-0">{iconRight}</span> : null}
    </button>
  );
}
