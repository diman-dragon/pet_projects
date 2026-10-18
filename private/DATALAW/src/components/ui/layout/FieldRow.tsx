// src/components/ui/layout/FieldRow.tsx
import type { ReactNode } from 'react';

interface FieldRowProps {
  label: string;
  value?: string | number | null;
  icon?: ReactNode;
  mono?: boolean;
  fallback?: string;
  className?: string;
}

export function FieldRow({
  label,
  value,
  icon,
  mono = false,
  fallback = '—',
  className = '',
}: FieldRowProps) {
  const isEmpty = value === undefined || value === null || value === '';

  return (
    <div className={['flex flex-col gap-0.5 py-1', className].join(' ')}>
      <span className="select-none text-[length:var(--text-xs)] font-medium leading-[var(--leading-body)] tracking-tight text-[var(--app-text-muted)]">
        {label}
      </span>

      <div className="flex min-h-[20px] items-center gap-2">
        {icon && (
          <span className="origin-left shrink-0 scale-90 text-[var(--app-text-muted)]">{icon}</span>
        )}

        <span
          className={[
            'text-[length:var(--text-sm)] leading-[var(--leading-heading)] transition-colors',
            mono
              ? 'font-mono font-medium tracking-tighter text-[var(--app-text-secondary)]'
              : 'font-semibold text-[var(--app-text)]',
            isEmpty ? 'font-normal italic text-[var(--app-text-muted)]' : '',
          ]
            .filter(Boolean)
            .join(' ')}
        >
          {isEmpty ? fallback : value}
        </span>
      </div>
    </div>
  );
}
