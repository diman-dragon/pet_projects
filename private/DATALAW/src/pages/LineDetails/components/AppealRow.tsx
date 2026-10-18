// src/pages/LineDetails/components/AppealRow.tsx

import { Edit2, Trash2, ChevronRight, FileText, Hash } from 'lucide-react';
import { Appeal } from '../types';
import { Button } from '../../../components/ui/form/Button';
import { Badge } from '../../../components/ui/data/Badge';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';

interface Props {
  appeal: Appeal;
  index: number;
  onEdit: (a: Appeal) => void;
  onDelete: (a: Appeal) => void;
  onNavigate: (a: Appeal) => void;
  compact?: boolean;
}

const STATUS_VARIANT: Record<string, 'blue' | 'green' | 'gray' | 'amber'> = {
  DRAFT: 'gray',
  READY: 'blue',
  SENT: 'amber',
  REGISTERED: 'blue',
  RECEIVED: 'green',
  CLOSED: 'green',
  ARCHIVE: 'gray',
  REJECTED: 'amber',
};

export function AppealRow({
  appeal: a,
  index,
  onEdit,
  onDelete,
  onNavigate,
  compact = false,
}: Props) {
  const { t } = useI18n();

  if (compact) {
    const statusKey = (a.status || 'DRAFT').toUpperCase();
    const variant = STATUS_VARIANT[statusKey] ?? STATUS_VARIANT.DRAFT;
    const trKey = `status.${statusKey.toLowerCase()}` as TranslationKey;
    const translated = t(trKey);
    const statusLabel = translated === trKey ? statusKey : translated;

    return (
      <tr
        className="group cursor-pointer transition-colors hover:bg-[var(--app-surface-soft)]"
        onClick={() => onNavigate(a)}
      >
        <td className="px-4 py-3">
          <span className="text-[13px] font-semibold text-[var(--app-text)] truncate">
            {a.title}
          </span>
        </td>
        <td className="px-4 py-3">
          <Badge label={statusLabel} variant={variant} />
        </td>
        <td className="px-4 py-3">
          <span className="text-[12px] text-[var(--app-text-secondary)]">
            {a.appeal_date ? new Date(a.appeal_date).toLocaleDateString() : '—'}
          </span>
        </td>
      </tr>
    );
  }

  const statusKey = (a.status || 'DRAFT').toUpperCase();
  const variant = STATUS_VARIANT[statusKey] ?? STATUS_VARIANT.DRAFT;
  const trKey = `status.${statusKey.toLowerCase()}` as TranslationKey;
  const translated = t(trKey);
  const statusLabel = translated === trKey ? statusKey : translated;

  return (
    <tr
      onClick={() => onNavigate(a)}
      className="group transition-colors cursor-pointer hover:bg-[var(--app-surface-soft)]"
    >
      <td className="py-3 pl-6 pr-3 font-mono text-[11px] font-bold text-[var(--app-text-muted)] tracking-tighter">
        {String(index + 1).padStart(2, '0')}
      </td>

      <td className="py-3 px-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-[10px] bg-[var(--app-surface-soft)] flex items-center justify-center text-[var(--app-text-muted)] group-hover:text-[var(--app-accent)] group-hover:bg-[var(--app-accent)]/[0.08] transition-all shrink-0">
            <FileText size={14} />
          </div>
          <span className="text-[13px] font-semibold text-[var(--app-text)] leading-none truncate">
            {a.title}
          </span>
        </div>
      </td>

      <td className="py-3 px-3">
        <div className="flex flex-col gap-0.5">
          <span className="text-[12px] text-[var(--app-text-secondary)] font-medium truncate">
            {a.agency_code || '—'}
          </span>
          <div className="flex items-center gap-1 text-[10px] text-[var(--app-text-muted)]">
            <Hash size={10} />
            <span className="truncate">{a.outgoing_number || t('appeals.row.noNumber')}</span>
          </div>
        </div>
      </td>

      <td className="py-3 px-3">
        <Badge label={statusLabel} variant={variant} />
      </td>

      <td className="py-3 pr-6 pl-3" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-all translate-x-2 group-hover:translate-x-0">
          <Button variant="ghost" size="sm" title={t('appeals.row.edit')} onClick={() => onEdit(a)}>
            <Edit2 size={13} />
          </Button>
          <Button variant="danger" size="sm" title={t('appeals.row.delete')} onClick={() => onDelete(a)}>
            <Trash2 size={13} />
          </Button>

          <div className="w-px h-4 bg-[var(--app-border)] mx-1" />

          <button
            type="button"
            onClick={() => onNavigate(a)}
            className="w-8 h-8 flex items-center justify-center rounded-[10px] bg-[var(--app-accent)] text-white shadow-sm hover:bg-[var(--app-accent-strong)] active:scale-95 transition-all"
          >
            <ChevronRight size={16} strokeWidth={3} />
          </button>
        </div>
      </td>
    </tr>
  );
}
