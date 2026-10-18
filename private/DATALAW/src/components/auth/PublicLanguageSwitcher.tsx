import { APP_LANGUAGE_LABELS, SUPPORTED_APP_LANGUAGES, type AppLanguage } from '../../i18n/config';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { UiCaption } from '../ui/atoms/UiCaption';

type PublicLanguageSwitcherProps = {
  className?: string;
  /** Компактный ряд без подписи-секции */
  compact?: boolean;
};

/**
 * Переключатель языка для экранов до входа в Shell (логин, setup, лицензия).
 * Пишет в localStorage через setLanguage (i18next + AppLanguageService).
 */
export function PublicLanguageSwitcher({
  className = '',
  compact = false,
}: PublicLanguageSwitcherProps) {
  const { t, language, setLanguage } = useI18n();

  return (
    <div
      className={['w-full max-w-[420px]', className].filter(Boolean).join(' ')}
      role="group"
      aria-label={t('auth.languageLabel')}
    >
      {!compact && (
        <UiCaption as="div" overline className="mb-2 text-center">
          {t('auth.languageLabel')}
        </UiCaption>
      )}
      <div className={`flex gap-1 ${compact ? 'justify-center' : ''}`}>
        {SUPPORTED_APP_LANGUAGES.map((lng: AppLanguage) => {
          const active = language === lng;
          return (
            <button
              key={lng}
              type="button"
              aria-pressed={active}
              title={APP_LANGUAGE_LABELS[lng]}
              onClick={() => {
                void setLanguage(lng);
              }}
              className={[
                'min-w-0 flex-1 rounded-md border px-2 py-2 text-center text-[length:var(--text-xs)] font-semibold leading-[var(--leading-heading)] transition-colors',
                active
                  ? 'border-[var(--app-accent)] bg-[color-mix(in_srgb,var(--app-accent)_12%,transparent)] text-[var(--app-accent)]'
                  : 'border-[var(--app-border)] bg-[var(--app-panel-bg)] text-[var(--app-text-secondary)] backdrop-blur-md hover:border-[var(--app-border-strong)] hover:bg-[color-mix(in_srgb,var(--app-text)_4%,transparent)]',
              ].join(' ')}
            >
              {lng.toUpperCase()}
            </button>
          );
        })}
      </div>
    </div>
  );
}
