import { ReactNode } from 'react';

export interface AppTableColumn {
  key: string;
  label: string;
  width?: string;
  align?: 'left' | 'center' | 'right';
}

interface AppTableProps {
  columns: AppTableColumn[];
  children: ReactNode;
  className?: string;
  elevated?: boolean;
  hideHeader?: boolean;
}

const alignClass: Record<NonNullable<AppTableColumn['align']>, string> = {
  left: 'text-left',
  center: 'text-center',
  right: 'text-right',
};

export function AppTable({
  columns,
  children,
  className = '',
  elevated = true,
  hideHeader = false,
}: AppTableProps) {
  return (
    <div
      className={[
        elevated
          ? 'overflow-hidden rounded-[var(--app-radius-md)] border border-[var(--app-border)] bg-[var(--app-surface)] shadow-[var(--app-shadow-sm)]'
          : 'overflow-hidden',
        className,
      ].join(' ')}
    >
      <table className="w-full table-fixed border-collapse">
        {!hideHeader && (
          <thead>
            <tr className="border-b border-[var(--app-border)] bg-[var(--app-surface-soft)]">
              {columns.map((column, index) => (
                <th
                  key={column.key}
                  style={column.width ? { width: column.width } : undefined}
                  className={[
                    'px-4 py-3 text-[length:var(--text-xs)] font-bold leading-[var(--leading-heading)] tracking-wide text-[var(--app-text-secondary)]',
                    index === 0 ? 'pl-6' : '',
                    index === columns.length - 1 ? 'pr-6' : '',
                    alignClass[column.align ?? 'left'],
                  ].join(' ')}
                >
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
        )}
        <tbody className="divide-y divide-[var(--app-border)] bg-[var(--app-surface)] [&>tr:hover]:bg-[color-mix(in_srgb,var(--app-accent)_5%,transparent)]">
          {children}
        </tbody>
      </table>
    </div>
  );
}
