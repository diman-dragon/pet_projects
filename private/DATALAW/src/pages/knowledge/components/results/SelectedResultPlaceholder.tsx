import { useI18n } from '../../../../i18n/AppLanguageProvider';

export function SelectedResultPlaceholder() {
  const { t } = useI18n();

  return (
    <div className="flex h-full items-center justify-center text-[var(--app-text-muted)]">
      <div className="text-center">
        <p className="mb-3 text-[length:var(--text-hero)] font-semibold">
          {t('knowledge.placeholder.fragmentTitle')}
        </p>
        <p className="text-[length:var(--text-sm)] leading-[var(--leading-body)]">
          {t('knowledge.placeholder.fragmentHint')}
        </p>
      </div>
    </div>
  );
}
