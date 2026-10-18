import { AlertTriangle } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';

interface DeleteConfirmContentProps {
  entityLabel?: string;
  entityName: string;
  note?: string;
}

export function DeleteConfirmContent({ entityLabel, entityName, note }: DeleteConfirmContentProps) {
  const { t } = useI18n();
  const resolvedEntity = entityLabel ?? t('delete.entityDefault');
  const label = resolvedEntity.toLowerCase();

  return (
    <div className="space-y-5">
      <div className="flex gap-4 rounded-[20px] border border-[var(--app-border)] bg-[var(--app-surface-soft)] p-4">
        <div className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-[14px] bg-[var(--app-warning-soft)] text-[var(--app-warning)]">
          <AlertTriangle size={20} strokeWidth={2.2} />
        </div>

        <div className="flex-1">
          <h3 className="text-[length:var(--text-md)] font-semibold leading-[var(--leading-heading)] tracking-[-0.02em] text-[var(--app-text)]">
            {t('delete.title', { entity: label })}
          </h3>
          <p className="mt-2 text-[length:var(--text-sm)] leading-[var(--leading-body)] text-[var(--app-text-secondary)]">
            {t('delete.body', {
              entity: label.charAt(0).toUpperCase() + label.slice(1),
              name: entityName,
            })}
          </p>
        </div>
      </div>

      <div className="rounded-[18px] border border-[var(--app-warning-border)] bg-[var(--app-warning-soft)] px-4 py-3 text-[length:var(--text-sm)] leading-[var(--leading-body)] text-[var(--app-warning)]">
        {note || t('delete.noteDefault')}
      </div>
    </div>
  );
}
