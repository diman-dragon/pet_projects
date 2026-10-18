import { useI18n } from '../../../i18n/AppLanguageProvider';

interface RequiredFieldsLegendProps {
  className?: string;
}

export function RequiredFieldsLegend({ className = '' }: RequiredFieldsLegendProps) {
  const { t } = useI18n();

  return (
    <div
      className={`rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-3.5 py-2.5 ${className}`.trim()}
    >
      <p className="text-[length:var(--text-xs)] leading-[var(--leading-body)] text-[var(--app-text-secondary)]">
        <span className="mr-1 font-black text-[var(--app-danger)]">*</span>
        {t('field.legendRequired')}
      </p>
    </div>
  );
}
