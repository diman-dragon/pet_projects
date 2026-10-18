import { PublicLanguageSwitcher } from '../../components/auth/PublicLanguageSwitcher';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { UiText } from '../../components/ui/atoms/UiText';
import { LicenseActivationPanel } from './components/LicenseActivationPanel';
import { LicensePageBackground } from './components/LicensePageBackground';
import { LicensePageHeader } from './components/LicensePageHeader';
import { LicenseTierPanel } from './components/LicenseTierPanel';

interface ActivationPageProps {
  onActivated: () => void;
}

export default function ActivationPage({ onActivated }: ActivationPageProps) {
  const { t } = useI18n();

  return (
    <div className="relative flex h-screen w-screen flex-col items-center justify-center overflow-hidden px-6">
      <LicensePageBackground />

      <div className="absolute right-6 top-6 z-20 w-full max-w-[min(100%,280px)] sm:right-8 sm:top-8">
        <PublicLanguageSwitcher compact />
      </div>

      <LicensePageHeader />

      <div className="relative z-10 grid w-full max-w-5xl gap-5 pb-8 lg:grid-cols-[1.1fr_0.9fr]">
        <LicenseActivationPanel onActivated={onActivated} />
        <LicenseTierPanel />
      </div>

      <div className="relative z-10 pb-4">
        <UiText
          as="p"
          size="xs"
          weight="semibold"
          tone="muted"
          heading
          className="text-center uppercase tracking-[0.22em]"
        >
          {t('welcome.footerBrand')} · {t('licensePage.footer')}
        </UiText>
      </div>
    </div>
  );
}
