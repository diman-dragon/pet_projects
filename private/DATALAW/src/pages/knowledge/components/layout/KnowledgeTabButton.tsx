import type { ReactNode } from 'react';

interface KnowledgeTabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: ReactNode;
  label: string;
  badge?: number;
}

export function KnowledgeTabButton({
  active,
  onClick,
  icon,
  label,
  badge,
}: KnowledgeTabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={[
        'flex items-center gap-1.5 border-b-2 px-4 py-2 text-[length:var(--text-sm)] font-medium transition-colors',
        active
          ? 'border-[var(--app-accent)] text-[var(--app-accent)]'
          : 'border-transparent text-[var(--app-text-secondary)] hover:text-[var(--app-text)]',
      ].join(' ')}
    >
      {icon}
      {label}
      {badge !== undefined && badge > 0 && (
        <span className="ml-1 rounded-full bg-[var(--app-border)] px-1.5 py-0.5 text-[length:var(--text-xs)] text-[var(--app-text-secondary)]">
          {badge}
        </span>
      )}
    </button>
  );
}
