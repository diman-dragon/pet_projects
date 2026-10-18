import { Globe2, Languages } from 'lucide-react';
import { Button } from '../../../components/ui/form/Button';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { SUPPORTED_APP_LANGUAGES, type AppLanguage } from '../../../i18n/config';

interface LanguageSettingsSectionProps {
  language: AppLanguage;
  systemLanguageLabel: string;
  onSetLanguage: (language: AppLanguage) => Promise<void>;
  onResetToSystem: () => Promise<void>;
}

export function LanguageSettingsSection({
  language,
  systemLanguageLabel,
  onSetLanguage,
  onResetToSystem,
}: LanguageSettingsSectionProps) {
  const { t } = useI18n();

  return (
    <section className="overflow-hidden rounded-[24px] border border-[var(--app-border)] bg-[var(--app-surface-raised)] shadow-[var(--app-shadow-sm)]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--app-border)] px-6 py-5">
        <div>
          <h2 className="flex items-center gap-2 text-[12px] font-semibold uppercase tracking-[0.22em] text-[var(--app-text-muted)]">
            <Languages size={14} />
            {t('settings.language.title')}
          </h2>
          <p className="mt-2 max-w-2xl text-[13px] leading-relaxed text-[var(--app-text-secondary)]">
            {t('settings.language.help')}
          </p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          icon={<Globe2 size={14} />}
          onClick={() => void onResetToSystem()}
        >
          {t('settings.language.followSystem')}
        </Button>
      </div>

      <div className="grid gap-5 px-6 py-6 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="block">
          <span className="mb-2 block text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--app-text-muted)]">
            {t('settings.language.label')}
          </span>
          <div
            className="flex flex-wrap gap-2 rounded-[18px] border border-[var(--app-border)] bg-[var(--app-surface-soft)] p-2"
            role="group"
            aria-label={t('settings.language.title')}
          >
            {SUPPORTED_APP_LANGUAGES.map((code) => {
              const active = language === code;
              return (
                <button
                  key={code}
                  type="button"
                  onClick={() => void onSetLanguage(code)}
                  className={`min-h-12 min-w-[4.5rem] flex-1 rounded-[14px] px-4 text-[14px] font-bold tracking-wide transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)] ${
                    active
                      ? 'bg-[var(--app-accent)] text-white shadow-[var(--app-shadow-sm)]'
                      : 'bg-transparent text-[var(--app-text-secondary)] hover:bg-[var(--app-surface-raised)] hover:text-[var(--app-text)]'
                  }`}
                >
                  {code.toUpperCase()}
                </button>
              );
            })}
          </div>
        </div>

        <div className="rounded-[20px] border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-5 py-4">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--app-text-muted)]">
            {t('settings.language.systemLabel')}
          </div>
          <div className="mt-2 text-[16px] font-semibold text-[var(--app-text)]">
            {systemLanguageLabel}
          </div>
          <div className="mt-1 text-[12px] leading-relaxed text-[var(--app-text-secondary)]">
            {t('settings.language.detected')}
          </div>
        </div>
      </div>
    </section>
  );
}
