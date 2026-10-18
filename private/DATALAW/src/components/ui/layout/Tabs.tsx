// src/components/ui/layout/Tabs.tsx
import type { ReactNode } from 'react';

export interface TabItem {
  id: string;
  label: string;
  icon?: ReactNode;
  count?: number;
}

interface TabsProps {
  items: TabItem[];
  active: string;
  onChange: (id: string) => void;
  className?: string;
}

export function Tabs({ items, active, onChange, className = '' }: TabsProps) {
  return (
    <div className={['relative flex items-center gap-1', className].join(' ')}>
      {items.map((tab) => {
        const isActive = tab.id === active;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onChange(tab.id)}
            className={[
              'group relative flex h-10 select-none items-center gap-2 rounded-md px-3 outline-none transition-all duration-200',
              'text-[length:var(--text-sm)] font-medium leading-[var(--leading-heading)] tracking-tight',
              isActive
                ? 'text-black'
                : 'text-black/50 hover:text-black hover:bg-black/[0.03] active:scale-95',
            ].join(' ')}
          >
            {tab.icon && (
              <span
                className={[
                  'shrink-0 transition-colors duration-200',
                  isActive ? 'text-[var(--app-accent)]' : 'text-black/30 group-hover:text-black/50',
                ].join(' ')}
              >
                {tab.icon}
              </span>
            )}

            <span>{tab.label}</span>

            {tab.count !== undefined && tab.count > 0 && (
              <span
                className={[
                  'ml-0.5 flex h-[18px] min-w-[18px] items-center justify-center rounded-full px-1.5 text-[length:var(--text-xs)] font-bold leading-[var(--leading-heading)] transition-all duration-200',
                  isActive
                    ? 'bg-[var(--app-accent)] text-white shadow-sm shadow-[rgba(0,120,212,0.3)]'
                    : 'bg-black/10 text-black/40 group-hover:bg-black/15',
                ].join(' ')}
              >
                {tab.count}
              </span>
            )}

            <div
              className={[
                'absolute bottom-[2px] left-3 right-3 h-[3px] rounded-full bg-[var(--app-accent)] transition-all duration-300 ease-out shadow-[0_1px_4px_rgba(0,120,212,0.3)]',
                isActive ? 'scale-x-100 opacity-100' : 'scale-x-0 opacity-0',
              ].join(' ')}
            />
          </button>
        );
      })}
    </div>
  );
}
