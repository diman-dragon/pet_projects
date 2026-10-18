// src/pages/license/components/LicenseFingerprintBox.tsx
//
// Блок отображения машинного отпечатка с кнопкой копирования.
// Fingerprint загружается автоматически при монтировании.

import { useEffect, useState } from 'react';
import { CheckCircle2, Copy, Loader2 } from 'lucide-react';
import { MachineFingerprintService } from '../../../services/license/MachineFingerprint';
import { useI18n } from '../../../i18n/AppLanguageProvider';

interface LicenseFingerprintBoxProps {
  onFingerprintLoaded?: (fp: string) => void;
}

export function LicenseFingerprintBox({ onFingerprintLoaded }: LicenseFingerprintBoxProps) {
  const { t } = useI18n();
  const [fingerprint, setFingerprint] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    MachineFingerprintService.get()
      .then((m) => {
        setFingerprint(m.fingerprint);
        onFingerprintLoaded?.(m.fingerprint);
      })
      .catch(() => setFingerprint(null))
      .finally(() => setLoading(false));
  }, []);

  const handleCopy = async () => {
    if (!fingerprint) return;
    await navigator.clipboard.writeText(fingerprint);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rounded-lg border border-black/[0.08] bg-black/[0.02] p-3">
      <div className="text-[9px] font-black uppercase tracking-[0.25em] text-black/30 mb-2">
        Machine Fingerprint
      </div>

      <div className="flex items-center gap-2">
        <div className="flex-1 font-mono text-[10px] text-black/50 break-all leading-relaxed min-h-[2rem] flex items-center">
          {loading ? (
            <span className="flex items-center gap-1.5 text-black/30">
              <Loader2 size={11} className="animate-spin" />
              {t('app.checking')}
            </span>
          ) : (
            (fingerprint ?? (
              <span className="text-red-400/70">{t('license.fingerprint.error')}</span>
            ))
          )}
        </div>

        {fingerprint && (
          <button
            onClick={handleCopy}
            className="shrink-0 flex items-center gap-1 px-2.5 py-1.5 rounded border border-black/[0.08] bg-white hover:bg-black/[0.02] text-[10px] font-bold text-black/50 transition-all"
          >
            {copied ? (
              <>
                <CheckCircle2 size={11} className="text-emerald-500" />{' '}
                {t('license.fingerprint.copied')}
              </>
            ) : (
              <>
                <Copy size={11} /> {t('license.fingerprint.copy')}
              </>
            )}
          </button>
        )}
      </div>

      <p className="text-[9px] text-black/25 mt-2 leading-relaxed">
        Сообщите этот код при получении лицензионного ключа.
      </p>
    </div>
  );
}
