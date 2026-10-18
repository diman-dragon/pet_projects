import { useState } from 'react';
import { KeyRound, Loader2, ShieldAlert } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { UiText } from '../../../components/ui/atoms/UiText';
import { Button } from '../../../components/ui/form/Button';
import { LicenseService } from '../../../services/license/LicenseService';
import { useLicenseStore } from '../../../store/useLicenseStore';
import { LicenseFingerprintBox } from './LicenseFingerprintBox';
import { LicenseKeyInput } from './LicenseKeyInput';

interface LicenseActivationPanelProps {
  onActivated: () => void;
}

export function LicenseActivationPanel({ onActivated }: LicenseActivationPanelProps) {
  const { t } = useI18n();
  const [licenseKey, setLicenseKey] = useState('');
  const [busy, setBusy] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const setLicense = useLicenseStore((s) => s.setLicense);

  const handleActivate = async () => {
    const key = licenseKey.trim();
    if (!key || busy || success) return;

    setBusy(true);
    setError(null);
    setSuccess(false);

    try {
      const license = await LicenseService.activate(key);
      setLicense(license);
      setSuccess(true);

      // Даем пользователю увидеть сообщение об успехе перед редиректом
      setTimeout(() => {
        onActivated();
      }, 1500);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t('licensePage.activateInvalid'));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-5 rounded-[24px] border border-[var(--app-border)] bg-[var(--app-surface-raised)] p-6 shadow-[var(--app-shadow-md)]">
      <div className="flex items-center gap-3 border-b border-[var(--app-border)] pb-4">
        <div
          className={`flex size-10 items-center justify-center rounded-[14px] ${success ? 'bg-[var(--app-success)]' : 'bg-[var(--app-accent)]'} text-[var(--app-surface)] shadow-[var(--app-shadow-sm)] transition-colors`}
        >
          <KeyRound size={17} />
        </div>
        <div>
          <UiText as="p" size="md" weight="semibold" tone="default" heading>
            {success ? t('licensePage.activateSuccess') : t('licensePage.activateTitle')}
          </UiText>
          <UiText as="p" size="xs" tone="secondary" className="mt-1 leading-[var(--leading-body)]">
            {success ? t('welcome.footerBrand') : t('licensePage.activateSubtitle')}
          </UiText>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <LicenseKeyInput
          value={licenseKey}
          onChange={setLicenseKey}
          onSubmit={handleActivate}
          disabled={busy || success}
        />
        <LicenseFingerprintBox />
      </div>

      {success ? (
        <div className="flex items-start gap-2.5 rounded-[16px] border border-[var(--app-success-border)] bg-[var(--app-success-soft)] px-3 py-3">
          <KeyRound size={14} className="mt-0.5 shrink-0 text-[var(--app-success)]" />
          <UiText
            as="p"
            size="xs"
            tone="default"
            className="font-medium leading-[var(--leading-body)] text-[var(--app-success)]"
          >
            {t('licensePage.activateSuccess')}
          </UiText>
        </div>
      ) : null}

      {error ? (
        <div className="flex items-start gap-2.5 rounded-[16px] border border-[color-mix(in_srgb,var(--app-danger)_18%,transparent)] bg-[color-mix(in_srgb,var(--app-danger)_10%,transparent)] px-3 py-3">
          <ShieldAlert size={14} className="mt-0.5 shrink-0 text-[var(--app-danger)]" />
          <UiText
            as="p"
            size="xs"
            tone="default"
            className="leading-[var(--leading-body)] text-[var(--app-danger)]"
          >
            {error}
          </UiText>
        </div>
      ) : null}

      <Button
        variant={success ? 'ghost' : 'primary'}
        size="md"
        onClick={handleActivate}
        disabled={busy || !licenseKey.trim() || success}
        icon={busy ? <Loader2 size={13} className="animate-spin" /> : <KeyRound size={13} />}
      >
        {busy
          ? t('licensePage.checking')
          : success
            ? t('documentUpload.done')
            : t('licensePage.activate')}
      </Button>

      <UiText as="p" size="xs" tone="muted" className="text-center leading-[var(--leading-body)]">
        {t('licensePage.activateHint')}
      </UiText>
    </div>
  );
}
