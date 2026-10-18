import { Shield } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';
import { TIER_FEATURES, type LicenseTier } from '../../../services/license/config';

const TIER_LABELS: Record<LicenseTier, string> = {
  basic: 'Basic',
  pro: 'Pro',
  enterprise: 'Enterprise',
};

const TIER_ACCENTS: Record<LicenseTier, string> = {
  basic: 'bg-slate-100 text-slate-700 border-slate-200',
  pro: 'bg-[var(--app-accent)]/[0.06] text-[var(--app-accent)] border-[var(--app-accent)]/20',
  enterprise: 'bg-amber-50 text-amber-700 border-amber-200',
};

export function LicenseTierPanel() {
  const { t } = useI18n();
  const tiers = Object.entries(TIER_FEATURES) as Array<
    [LicenseTier, (typeof TIER_FEATURES)[LicenseTier]]
  >;

  return (
    <div className="flex flex-col gap-4 rounded-[24px] border border-[var(--app-border)] bg-[var(--app-surface-raised)] p-6 shadow-[var(--app-shadow-sm)]">
      <div className="flex items-center gap-2 border-b border-[var(--app-border)] pb-4">
        <Shield size={14} className="text-[var(--app-text-muted)]" />
        <span className="text-[10px] font-semibold uppercase tracking-[0.22em] text-[var(--app-text-muted)]">
          {t('licensePage.tierTitle')}
        </span>
      </div>

      <div className="flex flex-col gap-3">
        {tiers.map(([tier, features]) => (
          <div key={tier} className={`rounded-[18px] border px-4 py-3 ${TIER_ACCENTS[tier]}`}>
            <div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.18em]">
              {TIER_LABELS[tier]}
            </div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(features).map(([feature, enabled]) => (
                <span
                  key={feature}
                  className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ${
                    enabled ? 'bg-white/75 text-black/70' : 'bg-black/[0.04] text-black/30'
                  }`}
                >
                  {t(`licensePage.feature.${feature}` as TranslationKey)}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <p className="mt-auto border-t border-[var(--app-border)] pt-3 text-[11px] leading-relaxed text-[var(--app-text-muted)]">
        {t('licensePage.tierFooter')}
      </p>
    </div>
  );
}
