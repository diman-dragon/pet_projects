// src/pages/ClientDetails/components/CasesTable.tsx

import { Edit2, Trash2, ChevronRight } from 'lucide-react';
import { Case } from '../../../types/entities';
import { Badge } from '../../../components/ui';
import { AppTable } from '../../../components/ui/data/AppTable';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';

interface CasesTableProps {
  cases: Case[];
  onEdit: (caseItem: Case) => void;
  onDelete: (caseItem: Case) => void;
  onNavigate: (caseItem: Case) => void;
}

const STATUS_VARIANT: Record<string, 'blue' | 'green' | 'amber' | 'gray'> = {
  ACTIVE: 'blue',
  PAUSED: 'amber',
  DONE: 'green',
  ARCHIVE: 'gray',
};

const CATEGORY_CONFIG: Record<string, { label: string; variant: 'blue' | 'green' | 'amber' | 'gray' }> = {
  litigation: { label: 'Судебное', variant: 'blue' },
  consulting: { label: 'Консультация', variant: 'amber' },
  contract: { label: 'Договор', variant: 'green' },
  other: { label: 'Прочее', variant: 'gray' },
};

function dateLocale(language: string): string {
  if (language === 'sr') return 'sr-Latn-RS';
  if (language === 'en') return 'en-US';
  return 'ru-RU';
}

export function CasesTable({ cases, onEdit, onDelete, onNavigate }: CasesTableProps) {
  const { t, language } = useI18n();
  const loc = dateLocale(language);

  const columns = [
    { key: 'id', label: 'ID', width: '5%' },
    { key: 'category', label: 'Категория', width: '12%' },
    { key: 'title', label: t('cases.table.col.title'), width: '30%' },
    { key: 'number', label: t('cases.table.col.number'), width: '15%' },
    { key: 'status', label: t('cases.table.col.status'), width: '12%' },
    { key: 'date', label: t('cases.table.col.date'), width: '15%' },
    { key: 'actions', label: t('cases.table.col.actions'), width: '11%', align: 'right' as const },
  ];

  return (
    <div className="w-full h-full min-h-0">
      <AppTable columns={columns} elevated={false} className="min-w-[1000px]">
        {cases.map((caseItem) => {
          const raw = caseItem.status?.toUpperCase() || 'ACTIVE';
          const variant = STATUS_VARIANT[raw] ?? STATUS_VARIANT.ACTIVE;
          const statusKey = `cases.status.${raw}` as TranslationKey;

          return (
            <tr
              key={caseItem.id}
              className="group cursor-pointer transition-colors hover:bg-[var(--app-surface-soft)]"
              onClick={() => onNavigate(caseItem)}
            >
              <td className="pl-6 pr-4 py-4">
                <span className="font-mono text-[11px] text-[var(--app-text-muted)]">
                  {String(caseItem.id).padStart(3, '0')}
                </span>
              </td>

              <td className="px-4 py-4">
                <Badge
                  label={CATEGORY_CONFIG[caseItem.case_type || 'other']?.label || 'Прочее'}
                  variant={CATEGORY_CONFIG[caseItem.case_type || 'other']?.variant || 'gray'}
                />
              </td>

              <td className="px-4 py-4">
                <div className="flex flex-col">
                  <span className="text-[13px] font-semibold text-[var(--app-text)] truncate">
                    {caseItem.title}
                  </span>
                  <div className="flex flex-col mt-0.5 gap-0.5">
                    {caseItem.case_type === 'litigation' && (
                      <span className="text-[11px] text-[var(--app-accent)] font-medium truncate">
                        {caseItem.litigation_subtype === 'civil'
                          ? '⚖️ Гражданское'
                          : caseItem.litigation_subtype === 'arbitration'
                            ? '⚖️ Арбитраж'
                            : caseItem.litigation_subtype === 'criminal'
                              ? '⚖️ Уголовное'
                              : caseItem.litigation_subtype === 'administrative'
                                ? '⚖️ Административное'
                                : '⚖️ Судебное'}
                      </span>
                    )}
                    {caseItem.summary && (
                      <span className="text-[10px] text-[var(--app-text-muted)] italic line-clamp-1 max-w-[300px]">
                        {caseItem.summary}
                      </span>
                    )}
                  </div>
                </div>
              </td>

              <td className="px-4 py-4">
                <span className="font-mono text-[12px] text-[var(--app-text-secondary)] tracking-tight">
                  {caseItem.case_number || '—'}
                </span>
              </td>

              <td className="px-4 py-4">
                <Badge label={t(statusKey)} variant={variant} />
              </td>

              <td className="px-4 py-4">
                <span className="text-[12px] text-[var(--app-text-secondary)]">
                  {caseItem.created_at
                    ? new Date(caseItem.created_at).toLocaleDateString(loc)
                    : '—'}
                </span>
              </td>

              <td className="pl-4 pr-6 py-4 text-right">
                <div
                  className="flex items-center justify-end gap-1"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onEdit(caseItem);
                    }}
                    className="w-8 h-8 flex items-center justify-center rounded-[10px] text-[var(--app-text-muted)] hover:text-[var(--app-text)] hover:bg-[var(--app-surface-soft)] transition-all"
                  >
                    <Edit2 size={15} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(caseItem);
                    }}
                    className="w-8 h-8 flex items-center justify-center rounded-[10px] text-[var(--app-text-muted)] hover:text-white hover:bg-[var(--app-danger)] transition-all"
                  >
                    <Trash2 size={15} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onNavigate(caseItem);
                    }}
                    className="w-8 h-8 flex items-center justify-center rounded-[10px] text-[var(--app-text-muted)] hover:text-[var(--app-text)] hover:bg-[var(--app-surface-soft)] transition-all"
                  >
                    <ChevronRight size={16} strokeWidth={2.5} />
                  </button>
                </div>
              </td>
            </tr>
          );
        })}
      </AppTable>

      {cases.length === 0 && (
        <div className="py-20 text-center text-[var(--app-text-muted)] text-[13px]">
          {t('cases.table.empty')}
        </div>
      )}
    </div>
  );
}
