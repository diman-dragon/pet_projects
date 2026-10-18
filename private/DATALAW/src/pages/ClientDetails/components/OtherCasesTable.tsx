import { Edit2, Trash2, ChevronRight } from 'lucide-react';
import { Case } from '../../../types/entities';
import { Badge } from '../../../components/ui';
import { AppTable } from '../../../components/ui/data/AppTable';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';

interface OtherCasesTableProps {
  cases: Case[];
  onEdit: (caseItem: Case) => void;
  onDelete: (caseItem: Case) => void;
  onNavigate: (caseItem: Case) => void;
}

const STATUS_VARIANT: Record<string, 'blue' | 'green' | 'amber' | 'gray'> = {
  LEAD: 'blue',
  ACTIVE: 'blue',
  PAUSED: 'amber',
  ARCHIVE: 'gray',
};

function dateLocale(language: string): string {
  if (language === 'sr') return 'sr-Latn-RS';
  if (language === 'en') return 'en-US';
  return 'ru-RU';
}

export function OtherCasesTable({ cases, onEdit, onDelete, onNavigate }: OtherCasesTableProps) {
  const { t, language } = useI18n();
  const loc = dateLocale(language);

  const columns = [
    { key: 'title', label: t('cases.table.col.title'), width: '40%' },
    { key: 'status', label: t('cases.table.col.status'), width: '20%' },
    { key: 'date', label: t('cases.table.col.date'), width: '25%' },
    { key: 'actions', label: t('cases.table.col.actions'), width: '15%', align: 'right' as const },
  ];

  return (
    <div className="w-full h-full min-h-0">
      <AppTable columns={columns} elevated={false} className="min-w-0 w-full table-fixed">
        {cases.map((caseItem) => {
          const raw = caseItem.status?.toUpperCase() || 'ACTIVE';
          const variant = STATUS_VARIANT[raw] ?? STATUS_VARIANT.ACTIVE;
          const statusKey = `case.status.${raw}` as TranslationKey;

          return (
            <tr
              key={caseItem.id}
              className="group cursor-pointer transition-colors hover:bg-[var(--app-surface-soft)]"
              onClick={() => onNavigate(caseItem)}
            >
              <td className="pl-6 pr-4 py-4">
                <div className="flex flex-col">
                  <span className="text-[13px] font-semibold text-[var(--app-text)] truncate">
                    {caseItem.title}
                  </span>
                  {caseItem.summary && (
                    <span className="text-[10px] text-[var(--app-text-muted)] italic line-clamp-1 mt-0.5">
                      {caseItem.summary}
                    </span>
                  )}
                </div>
              </td>

              <td className="px-4 py-4">
                <Badge label={t(statusKey)} variant={variant} />
              </td>

              <td className="px-4 py-4">
                <span className="text-[12px] text-[var(--app-text-secondary)]">
                  {caseItem.created_at ? new Date(caseItem.created_at).toLocaleDateString(loc) : '—'}
                </span>
              </td>

              <td className="pl-4 pr-6 py-4 text-right">
                <div className="flex items-center justify-end gap-1" onClick={(e) => e.stopPropagation()}>
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
