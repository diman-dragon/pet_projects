import { DorlaneLogo } from '../../../components/common/DorlaneLogo';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { UiText } from '../../../components/ui/atoms/UiText';

export function LicensePageHeader() {
  const year = new Date().getFullYear();
  const { t } = useI18n();

  return (
    <div className="relative z-10 flex flex-col items-center gap-4 pb-6 pt-8">
      <div className="flex items-center gap-3">
        <div className="h-px w-16 bg-[color-mix(in_srgb,var(--app-text)_15%,transparent)]" />
        <UiText
          as="span"
          size="xs"
          weight="semibold"
          tone="muted"
          heading
          className="uppercase tracking-[0.28em]"
        >
          {t('licensePage.headerKicker')} · {year}
        </UiText>
        <div className="h-px w-16 bg-[color-mix(in_srgb,var(--app-text)_15%,transparent)]" />
      </div>

      <DorlaneLogo size="md" showText />

      <div className="rounded-full border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-4 py-2">
        <UiText
          as="span"
          size="xs"
          weight="semibold"
          tone="secondary"
          heading
          className="uppercase tracking-[0.22em]"
        >
          {t('licensePage.headerBadge')}
        </UiText>
      </div>
    </div>
  );
}
