// src/pages/ClientDetails/components/ContractsTable.tsx

import { Edit2, Trash2, ChevronRight } from 'lucide-react';
import { Case } from '../../../types/entities';
import { Badge } from '../../../components/ui';
import { AppTable } from '../../../components/ui/data/AppTable';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';

interface ContractsTableProps {
  cases: Case[];
  onEdit: (caseItem: Case) => void;
  onDelete: (caseItem: Case) => void;
  onNavigate: (caseItem: Case) => void;
}

const STATUS_VARIANT: Record<string, 'blue' | 'green' | 'amber' | 'gray' | 'red'> = {
  ACTIVE: 'blue',
  DONE: 'green',
  TERMINATED: 'red',
  PAUSED: 'amber',
  ARCHIVE: 'gray',
};

function dateLocale(language: string): string {
  if (language === 'sr') return 'sr-Latn-RS';
  if (language === 'en') return 'en-US';
  return 'ru-RU';
}

export function ContractsTable({ cases, onEdit, onDelete, onNavigate }: ContractsTableProps) {
  const { t, language } = useI18n();
  const loc = dateLocale(language);

  const columns = [
    { key: 'counterparty', label: t('case.field.counterparty' as TranslationKey), width: '25%' },
    { key: 'number', label: t('case.field.contractNumber' as TranslationKey), width: '20%' },
    { key: 'date', label: t('case.field.contractDate' as TranslationKey), width: '20%' },
    { key: 'status', label: t('cases.table.col.status' as TranslationKey), width: '20%' },
    { key: 'actions', label: t('cases.table.col.actions' as TranslationKey), width: '15%', align: 'right' as const },
  ];

  return (
    <div className="w-full h-full min-h-0">
      <AppTable columns={columns} elevated={false} className="min-w-[1000px]">
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
                <span className="text-[13px] font-semibold text-[var(--app-text)] truncate">
                  {caseItem.counterparty || t('line.field.notSpecified' as TranslationKey)}
                </span>
              </td>

              <td className="px-4 py-4">
                <span className="font-mono text-[12px] text-[var(--app-text-secondary)] tracking-tight">
                  {caseItem.case_number || 'NOT_ENT'}
                </span>
              </td>

              <td className="px-4 py-4">
                <span className="text-[12px] text-[var(--app-text-secondary)]">
                  {caseItem.contract_date
                    ? new Date(caseItem.contract_date).toLocaleDateString(loc)
                    : new Date().toLocaleDateString(loc)}
                </span>
              </td>

              <td className="px-4 py-4">
                <Badge label={t(statusKey)} variant={variant} />
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
          {t('cases.table.empty' as TranslationKey)}
        </div>
      )}
    </div>
  );
}
