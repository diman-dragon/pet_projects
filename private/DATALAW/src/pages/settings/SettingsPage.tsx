import { Sparkles } from 'lucide-react';
import { DorlaneLogo } from '../../components/common/DorlaneLogo';
import { UiPanel } from '../../components/ui/atoms/UiPanel';
import { UiText } from '../../components/ui/atoms/UiText';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { APP_LANGUAGE_LABELS } from '../../i18n/config';
import { PageBase } from '../../shared/ui/PageBase';
import { AppLanguageService } from '../../services/config/AppLanguageService';
import { LanguageSettingsSection } from './components/LanguageSettingsSection';
import { LicenseInfoSection } from './components/LicenseInfoSection';
import { OllamaSettingsSection } from './components/OllamaSettingsSection';

export default function SettingsPage() {
  const { language, setLanguage, resetLanguageToSystem, t } = useI18n();
  const systemLanguage = AppLanguageService.getSystemLanguage();
  const systemLanguageLabel = APP_LANGUAGE_LABELS[systemLanguage];

  return (
    <PageBase
      title={t('settings.title')}
      subtitle={t('settings.subtitle')}
      icon={<Sparkles size={18} />}
    >
      <div className="flex flex-col gap-6">
        <UiPanel padding="lg" mica elevation="md" className="rounded-[24px]">
          <div className="flex flex-wrap items-center justify-between gap-5">
            <div className="max-w-2xl">
              <UiText as="p" size="xs" weight="bold" tone="muted" className="label-caps">
                {t('settings.badgeLabel')}
              </UiText>
              <UiText as="h2" size="md" weight="semibold" heading className="mt-3">
                {t('settings.title')}
              </UiText>
              <UiText as="p" size="sm" tone="secondary" className="mt-2 max-w-xl">
                {t('settings.subtitle')}
              </UiText>
            </div>
            <UiPanel padding="md" mica elevation="sm" className="rounded-[22px] border-[var(--app-border)]">
              <div className="flex items-center gap-4">
                <DorlaneLogo size="sm" showText={false} />
                <div>
                  <UiText as="p" size="xs" weight="bold" tone="muted" className="label-caps">
                    {t('settings.heroCardProduct')}
                  </UiText>
                  <UiText as="p" size="sm" weight="medium" className="mt-1">
                    {t('settings.heroCardTagline')}
                  </UiText>
                </div>
              </div>
            </UiPanel>
          </div>
        </UiPanel>

        <LanguageSettingsSection
          language={language}
          systemLanguageLabel={systemLanguageLabel}
          onSetLanguage={setLanguage}
          onResetToSystem={resetLanguageToSystem}
        />
        <OllamaSettingsSection />
        <LicenseInfoSection />

        <UiPanel padding="md" mica elevation="sm" className="rounded-[22px]">
          <div className="flex items-center gap-4">
            <DorlaneLogo size="sm" showText={false} />
            <div>
              <UiText as="p" size="xs" weight="semibold">
                {t('settings.footer.title')}
              </UiText>
              <UiText as="p" size="xs" tone="secondary" className="mt-1 max-w-xl">
                {t('settings.footer.text')}
              </UiText>
            </div>
          </div>
        </UiPanel>
      </div>
    </PageBase>
  );
}
