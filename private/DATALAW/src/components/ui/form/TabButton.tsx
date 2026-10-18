// src/components/ui/form/TabButton.tsx
import type { ReactNode } from 'react';

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  icon?: ReactNode;
  label: string;
  badge?: number;
}

export function TabButton({ active, onClick, icon, label, badge }: TabButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'group relative flex h-9 select-none items-center gap-2 rounded-md px-3 outline-none transition-all duration-200',
        'text-[length:var(--text-sm)] font-semibold leading-[var(--leading-heading)] tracking-tight',
        active
          ? 'bg-[color-mix(in_srgb,var(--app-text)_3%,transparent)] text-[var(--app-text)]'
          : 'text-[var(--app-text-muted)] hover:bg-[color-mix(in_srgb,var(--app-text)_2%,transparent)] hover:text-[var(--app-text-secondary)] active:scale-95',
      ].join(' ')}
    >
      {icon && (
        <span
          className={[
            'shrink-0 transition-colors',
            active
              ? 'text-[var(--app-accent)]'
              : 'text-[var(--app-text-muted)] group-hover:text-[var(--app-text-secondary)]',
          ].join(' ')}
        >
          {icon}
        </span>
      )}

      <span>{label}</span>

      {badge !== undefined && badge > 0 && (
        <span
          className={[
            'ml-0.5 flex h-[18px] min-w-[18px] items-center justify-center rounded-full px-1.5 text-[length:var(--text-xs)] font-bold leading-[var(--leading-heading)] transition-colors',
            active
              ? 'bg-[var(--app-accent)] text-[var(--app-surface-raised)] shadow-sm shadow-[color-mix(in_srgb,var(--app-accent)_35%,transparent)]'
              : 'bg-[color-mix(in_srgb,var(--app-text)_8%,transparent)] text-[var(--app-text-muted)]',
          ].join(' ')}
        >
          {badge}
        </span>
      )}

      <div
        className={[
          'absolute bottom-1 left-3 right-3 h-[2.5px] rounded-full bg-[var(--app-accent)] transition-all duration-300',
          active ? 'translate-y-0 opacity-100' : 'translate-y-1 opacity-0',
        ].join(' ')}
      />
    </button>
  );
}
