import type { ReactNode } from 'react';

export type BadgeVariant =
  | 'default'
  | 'success'
  | 'warning'
  | 'danger'
  | 'info'
  | 'muted'
  | 'blue'
  | 'green'
  | 'amber'
  | 'gray'
  | 'violet'
  | 'red'
  | 'purple';
export type BadgeSize = 'sm' | 'md';

const variantClass: Record<BadgeVariant, string> = {
  default:
    'border-[var(--app-border)] bg-[var(--app-surface-soft)] text-[var(--app-text-secondary)]',
  success:
    'border-[color-mix(in_srgb,var(--app-success)_22%,transparent)] bg-[color-mix(in_srgb,var(--app-success)_15%,transparent)] text-[var(--app-success)]',
  warning:
    'border-[color-mix(in_srgb,var(--app-warning)_22%,transparent)] bg-[color-mix(in_srgb,var(--app-warning)_15%,transparent)] text-[var(--app-warning)]',
  danger:
    'border-[color-mix(in_srgb,var(--app-danger)_22%,transparent)] bg-[color-mix(in_srgb,var(--app-danger)_15%,transparent)] text-[var(--app-danger)]',
  info:
    'border-[color-mix(in_srgb,var(--app-info)_22%,transparent)] bg-[color-mix(in_srgb,var(--app-info)_15%,transparent)] text-[var(--app-info)]',
  muted:
    'border-[var(--app-border)] bg-[color-mix(in_srgb,var(--app-text)_4%,transparent)] text-[var(--app-text-muted)]',
  blue:
    'border-[color-mix(in_srgb,var(--app-info)_22%,transparent)] bg-[color-mix(in_srgb,var(--app-info)_15%,transparent)] text-[var(--app-info)]',
  green:
    'border-[color-mix(in_srgb,var(--app-success)_22%,transparent)] bg-[color-mix(in_srgb,var(--app-success)_15%,transparent)] text-[var(--app-success)]',
  amber:
    'border-[color-mix(in_srgb,var(--app-warning)_22%,transparent)] bg-[color-mix(in_srgb,var(--app-warning)_15%,transparent)] text-[var(--app-warning)]',
  gray:
    'border-[var(--app-border)] bg-[color-mix(in_srgb,var(--app-text)_4%,transparent)] text-[var(--app-text-muted)]',
  violet:
    'border-[color-mix(in_srgb,var(--app-accent)_22%,transparent)] bg-[color-mix(in_srgb,var(--app-accent)_12%,transparent)] text-[var(--app-accent)]',
  red:
    'border-[color-mix(in_srgb,var(--app-danger)_22%,transparent)] bg-[color-mix(in_srgb,var(--app-danger)_15%,transparent)] text-[var(--app-danger)]',
  purple:
    'border-[color-mix(in_srgb,var(--app-accent)_22%,transparent)] bg-[color-mix(in_srgb,var(--app-accent)_12%,transparent)] text-[var(--app-accent)]',
};

const sizeClass: Record<BadgeSize, string> = {
  sm: 'rounded-[var(--app-radius-sm)] px-2 py-0.5 text-[length:var(--text-xs)]',
  md: 'rounded-[var(--app-radius-md)] px-2.5 py-1 text-[length:var(--text-xs)]',
};

interface BadgeProps {
  label: string;
  variant?: BadgeVariant;
  size?: BadgeSize;
  icon?: ReactNode;
  className?: string;
}

interface DotProps {
  variant?: BadgeVariant;
  className?: string;
}

export function StatusDot({ variant = 'muted', className = '' }: DotProps) {
  const toneClass =
    variant === 'success' || variant === 'green'
      ? 'status-dot status-dot--green'
      : variant === 'danger' || variant === 'red'
        ? 'status-dot status-dot--red'
        : variant === 'warning' || variant === 'amber'
          ? 'status-dot status-dot--yellow'
          : 'status-dot status-dot--muted';

  return <span className={[toneClass, className].join(' ')} />;
}

export function Badge({
  label,
  variant = 'default',
  size = 'sm',
  icon,
  className = '',
}: BadgeProps) {
  return (
    <span
      className={[
        'inline-flex items-center gap-1.5 border font-semibold leading-[var(--leading-heading)] whitespace-nowrap',
        variantClass[variant],
        sizeClass[size],
        className,
      ].join(' ')}
    >
      {icon ? <span className="shrink-0">{icon}</span> : null}
      {label}
    </span>
  );
}
