// src/pages/audit/components/DbTab.tsx

import { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Link as LinkIcon,
  Table as TableIcon,
} from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TableStructureWithCount } from '../../../services/db/InspectorService';
import type { DbInfo } from '../types';

const DB_COLOURS: Record<string, string> = {
  clients_main: '#0067c0',
  user_system: '#7c3aed',
  kb_main: '#059669',
};

function dbColour(key: string) {
  return DB_COLOURS[key] ?? '#64748b';
}

function TableCard({ table }: { table: TableStructureWithCount }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(true);

  return (
    <div className="overflow-hidden rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-raised)] shadow-[var(--app-shadow-sm)]">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between border-b border-[var(--app-border)] bg-[var(--app-surface-soft)] px-4 py-3 transition-colors hover:bg-[color-mix(in_srgb,var(--app-text)_3%,transparent)]"
      >
        <div className="flex items-center gap-2">
          <TableIcon className="h-4 w-4 text-[var(--app-text-muted)]" />
          <span className="text-[length:var(--text-sm)] font-bold leading-[var(--leading-heading)] text-[var(--app-text)]">
            {table.name}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="rounded bg-[var(--app-accent)] px-2 py-0.5 text-[length:var(--text-xs)] font-bold leading-[var(--leading-heading)] text-white">
            {t('audit.db.rowsLabel', { count: table.rowCount })}
          </span>
          {open ? (
            <ChevronUp size={14} className="text-[var(--app-text-muted)]" />
          ) : (
            <ChevronDown size={14} className="text-[var(--app-text-muted)]" />
          )}
        </div>
      </button>
      {open && (
        <div className="space-y-2 p-4">
          {table.columns.map((column) => (
            <div
              key={column.name}
              className="flex items-center justify-between text-[length:var(--text-xs)] leading-[var(--leading-body)]"
            >
              <div className="flex items-center gap-2 overflow-hidden">
                <div
                  className={`size-1.5 shrink-0 rounded-full ${column.isPk ? 'bg-orange-500' : 'bg-[color-mix(in_srgb,var(--app-text)_10%,transparent)]'}`}
                />
                <span
                  className={`truncate ${column.isPk ? 'font-bold text-[var(--app-text)]' : 'text-[var(--app-text-secondary)]'}`}
                >
                  {column.name}
                </span>
              </div>
              <span className="ml-2 shrink-0 rounded bg-[color-mix(in_srgb,var(--app-text)_4%,transparent)] px-1.5 py-0.5 font-mono text-[length:var(--text-xs)] font-bold uppercase leading-[var(--leading-heading)] text-[var(--app-text-muted)]">
                {column.type || t('audit.db.typeAny')}
              </span>
            </div>
          ))}
          {table.foreignKeys.map((foreignKey, index) => (
            <div
              key={index}
              className="mt-1 flex items-center gap-2 rounded border border-[color-mix(in_srgb,#6366f1_20%,var(--app-border))] bg-[color-mix(in_srgb,#6366f1_6%,transparent)] p-2 text-[length:var(--text-xs)] leading-[var(--leading-body)] text-indigo-700/90"
            >
              <LinkIcon className="h-3 w-3 shrink-0" />
              <span className="truncate">{foreignKey.from}</span>
              <ChevronRight className="h-3 w-3 shrink-0 opacity-30" />
              <span className="truncate underline decoration-indigo-200">
                {foreignKey.table}.{foreignKey.to}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function DbTab({ databases }: { databases: DbInfo[] }) {
  const { t, language } = useI18n();

  return (
    <div className="space-y-8">
      {databases.map((db) => {
        const colour = dbColour(db.key);
        const totalRows = db.tables.reduce((sum, table) => sum + table.rowCount, 0);
        const locale = language === 'en' ? 'en-US' : language === 'sr' ? 'sr-Latn-RS' : 'ru-RU';

        return (
          <div key={db.key}>
            <div className="mb-4 flex items-center gap-3">
              <div
                className="flex size-8 items-center justify-center rounded-lg"
                style={{ backgroundColor: `${colour}15` }}
              >
                <TableIcon size={16} style={{ color: colour }} />
              </div>
              <div>
                <span className="font-mono text-[length:var(--text-sm)] font-bold leading-[var(--leading-heading)] text-[var(--app-text)]">
                  {db.key}
                </span>
                <span className="ml-2 text-[length:var(--text-xs)] leading-[var(--leading-body)] text-[var(--app-text-muted)]">
                  {t('audit.db.tablesRows', {
                    tables: db.tables.length,
                    rows: totalRows.toLocaleString(locale),
                  })}
                </span>
              </div>
            </div>

            {db.tables.length === 0 ? (
              <p className="px-2 text-[length:var(--text-xs)] italic leading-[var(--leading-body)] text-[var(--app-text-muted)]">
                {t('audit.db.empty')}
              </p>
            ) : (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                {db.tables.map((table) => (
                  <TableCard key={table.name} table={table} />
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
