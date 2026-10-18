// src/components/ui/data/RegistryCard.tsx
import { ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';

interface RegistryCardProps {
  icon?: ReactNode;
  title: string;
  subtitle?: string;
  meta?: string;
  badge?: ReactNode;
  actions?: ReactNode;
  onClick?: () => void;
  className?: string;
}

export function RegistryCard({
  icon,
  title,
  subtitle,
  meta,
  badge,
  actions,
  onClick,
  className = '',
}: RegistryCardProps) {
  return (
    <div
      className={[
        'group relative flex items-center gap-4 px-4 py-3 mx-1 my-0.5 rounded-lg', // Отступы для эффекта выделения
        'border-b border-black/[0.03] last:border-0',
        'hover:bg-black/[0.025] hover:shadow-[inset_0_0_0_1px_rgba(0,0,0,0.02)] transition-all duration-200',
        onClick ? 'cursor-pointer active:scale-[0.99] active:bg-black/[0.04]' : '',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      onClick={onClick}
    >
      {/* Декоративный индикатор фокуса слева (Win11 Style) */}
      {onClick && (
        <div className="absolute left-1 w-1 h-4 rounded-full bg-blue-600 opacity-0 group-hover:opacity-100 transition-opacity" />
      )}

      {/* Иконка с мягким фоном */}
      {icon && (
        <div className="shrink-0 w-10 h-10 flex items-center justify-center rounded-xl bg-black/[0.03] text-black/40 group-hover:bg-white group-hover:shadow-sm transition-all duration-300">
          {icon}
        </div>
      )}

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-[14px] font-semibold text-black/85 truncate tracking-tight group-hover:text-black transition-colors">
            {title}
          </span>
          {badge && <div className="shrink-0 scale-90 origin-left">{badge}</div>}
        </div>
        {subtitle && (
          <span className="text-[12px] text-black/40 truncate block mt-0.5 leading-tight font-medium">
            {subtitle}
          </span>
        )}
      </div>

      <div className="flex items-center gap-4 shrink-0">
        {meta && (
          <span className="rounded border border-[var(--app-border)] bg-[color-mix(in_srgb,var(--app-text)_2%,transparent)] px-1.5 py-0.5 font-mono text-[length:var(--text-xs)] leading-[var(--leading-body)] text-[var(--app-text-muted)]">
            {meta}
          </span>
        )}

        {actions && (
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity translate-x-2 group-hover:translate-x-0 duration-200">
            {actions}
          </div>
        )}

        {onClick && !actions && (
          <ChevronRight
            size={16}
            strokeWidth={2.5}
            className="text-black/10 group-hover:text-black/30 transition-all transform group-hover:translate-x-0.5"
          />
        )}
      </div>
    </div>
  );
}
