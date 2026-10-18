// components/analyze/AnalyzeRisks.tsx

import { AlertTriangle } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

interface AnalyzeRisksProps {
  risks: string[];
}

/** Список рисков и замечаний */
export function AnalyzeRisks({ risks }: AnalyzeRisksProps) {
  const { t } = useI18n();

  return (
    <div>
      <div className="flex items-center gap-1.5 mb-2">
        <AlertTriangle size={11} className="text-amber-500" />
        <span className="text-[length:var(--text-xs)] font-black text-black/30 uppercase tracking-wider">
          {t('laila.analyze.section.risks')}
        </span>
      </div>
      <div className="space-y-1.5">
        {risks.map((risk, i) => (
          <div
            key={i}
            className="flex gap-2 items-start p-2.5 rounded-lg bg-amber-50/60 border border-amber-100"
          >
            <AlertTriangle size={10} className="text-amber-400 mt-0.5 shrink-0" />
            <p className="text-[length:var(--text-xs)] text-amber-800/70 leading-[var(--leading-body)]">
              {risk}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
