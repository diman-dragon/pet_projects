import { Appeal } from '../../../types/entities';
import { AppTable } from '../../../components/ui/data/AppTable';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { AppealRow } from './AppealRow';

interface Props {
  items: Appeal[];
  onEdit: (a: Appeal) => void;
  onDelete: (a: Appeal) => void;
  onNavigate: (a: Appeal) => void;
  compact?: boolean;
  hideHeader?: boolean;
}

export function AppealsTable({
  items,
  onEdit,
  onDelete,
  onNavigate,
  compact = false,
  hideHeader = false,
}: Props) {
  const { t } = useI18n();

  const columns = compact
    ? [
        { key: 'title', label: t('appeals.compact.col.title' as any), width: '40%' },
        { key: 'status', label: t('appeals.compact.col.status' as any), width: '30%' },
        { key: 'date', label: t('appeals.compact.col.date' as any), width: '30%' },
      ]
    : [
        { key: 'idx', label: '#', width: '50px' },
        { key: 'title', label: t('appeals.table.col.title'), width: '45%' },
        { key: 'agency', label: t('appeals.table.col.agency'), width: '25%' },
        { key: 'status', label: t('appeals.table.col.status'), width: '15%' },
        {
          key: 'actions',
          label: t('appeals.table.col.actions'),
          width: '15%',
          align: 'right' as const,
        },
      ];

  return (
    <div className="w-full h-full flex flex-col min-h-0">
      <div className="flex-1 overflow-auto min-h-0">
        <AppTable columns={columns} elevated={false} hideHeader={hideHeader} className="min-w-0 w-full table-fixed">
          {items.map((appeal, index) => (
            <AppealRow
              key={appeal.id}
              appeal={appeal}
              index={index}
              onEdit={onEdit}
              onDelete={onDelete}
              onNavigate={onNavigate}
              compact={compact}
            />
          ))}
        </AppTable>
      </div>

      {items.length === 0 && (
        <div className="py-20 text-center text-[var(--app-text-muted)] text-[13px] font-medium shrink-0">
          {t('appeals.table.empty')}
        </div>
      )}
    </div>
  );
}
