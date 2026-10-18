import { useState } from 'react';
import { CheckCircle2, RotateCcw, Shield, XCircle } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { Button } from '../../../components/ui/form/Button';
import { LicenseService } from '../../../services/license/LicenseService';
import { useLicenseStore } from '../../../store/useLicenseStore';
import { LicenseInfoRow } from './LicenseInfoRow';
import { LicenseResetConfirmModal } from './LicenseResetConfirmModal';

export function LicenseInfoSection() {
  const { language, t } = useI18n();
  const license = useLicenseStore((state) => state.license);
  const setLicense = useLicenseStore((state) => state.setLicense);
  const [showConfirm, setShowConfirm] = useState(false);
  const [resetting, setResetting] = useState(false);

  const handleReset = async () => {
    setResetting(true);
    await LicenseService.reset().catch(() => undefined);
    setLicense(null);
    setShowConfirm(false);
    window.location.reload();
  };

  const isValid = license?.valid === true;
  const isMaster = license?.isMaster === true;
  const expiresAt = license?.expiresAt ? new Date(license.expiresAt) : null;
  const isExpired = expiresAt ? expiresAt <= new Date() : false;
  const dateLocale = language === 'sr' ? 'sr-Latn-RS' : language === 'en' ? 'en-US' : 'ru-RU';

  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <Shield size={13} className="text-black/35" />
        <h2 className="text-[10px] font-black uppercase tracking-[0.25em] text-black/35">
          {t('license.section')}
        </h2>
      </div>

      <div className="overflow-hidden rounded-[18px] border border-black/[0.08] bg-white shadow-sm">
        <div
          className={`flex items-center gap-2.5 border-b border-black/[0.05] px-5 py-3 ${isValid && !isExpired ? 'bg-emerald-50' : 'bg-rose-50'}`}
        >
          {isValid && !isExpired ? (
            <>
              <CheckCircle2 size={14} className="text-emerald-500" />
              <span className="text-[11px] font-bold text-emerald-700">
                {isMaster ? t('license.master') : t('license.active')}
              </span>
            </>
          ) : (
            <>
              <XCircle size={14} className="text-rose-400" />
              <span className="text-[11px] font-bold text-rose-700">
                {isExpired ? t('license.expired') : t('license.missing')}
              </span>
            </>
          )}
        </div>

        <div className="divide-y divide-black/[0.04]">
          <LicenseInfoRow label={t('license.key')} value={license?.key ?? '—'} mono />
          <LicenseInfoRow label={t('license.tier')} value={license?.tier?.toUpperCase() ?? '—'} />
          <LicenseInfoRow
            label={t('license.expiresAt')}
            value={
              isMaster
                ? t('license.perpetual')
                : expiresAt
                  ? expiresAt.toLocaleDateString(dateLocale, {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric',
                    })
                  : '—'
            }
          />
          <LicenseInfoRow
            label={t('license.fingerprint')}
            value={license?.fingerprint ?? '—'}
            mono
          />
          <LicenseInfoRow
            label={t('license.issuedAt')}
            value={license?.issuedAt ? new Date(license.issuedAt).toLocaleDateString(dateLocale) : '—'}
          />
        </div>

        <div className="flex items-center justify-between gap-4 bg-[var(--app-surface-soft)] px-5 py-4">
          <p className="max-w-md text-[11px] leading-relaxed text-black/42">
            {t('license.resetHint')}
          </p>
          <Button
            variant="danger"
            size="sm"
            icon={<RotateCcw size={12} />}
            onClick={() => setShowConfirm(true)}
            disabled={resetting}
          >
            {t('license.resetAction')}
          </Button>
        </div>
      </div>

      <LicenseResetConfirmModal
        open={showConfirm}
        loading={resetting}
        onConfirm={handleReset}
        onCancel={() => setShowConfirm(false)}
      />
    </section>
  );
}
