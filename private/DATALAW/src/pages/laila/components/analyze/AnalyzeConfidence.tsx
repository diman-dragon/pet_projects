// components/analyze/AnalyzeConfidence.tsx

import { useI18n } from '../../../../i18n/AppLanguageProvider';

interface AnalyzeConfidenceProps {
  value: number; // 0–100
}

/** Полоса уверенности модели в анализе */
export function AnalyzeConfidence({ value }: AnalyzeConfidenceProps) {
  const { t } = useI18n();

  return (
    <div className="flex items-center justify-between p-3 rounded-xl bg-violet-50/60 border border-violet-100">
      <span className="text-[length:var(--text-xs)] font-black text-violet-600/70 uppercase tracking-wide">
        {t('laila.analyze.section.confidence')}
      </span>
      <div className="flex items-center gap-2">
        <div className="w-24 h-1.5 rounded-full bg-violet-100 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-violet-400 to-[#0067c0] rounded-full transition-all duration-1000"
            style={{ width: `${value}%` }}
          />
        </div>
        <span className="text-[length:var(--text-xs)] font-black text-violet-600">{value}%</span>
      </div>
    </div>
  );
}
