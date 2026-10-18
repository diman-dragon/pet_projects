// components/analyze/AnalyzeKeyPoints.tsx

import { CheckCircle2 } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

interface AnalyzeKeyPointsProps {
  points: string[];
}

/** Нумерованный список ключевых тезисов */
export function AnalyzeKeyPoints({ points }: AnalyzeKeyPointsProps) {
  const { t } = useI18n();

  return (
    <div>
      <div className="flex items-center gap-1.5 mb-2">
        <CheckCircle2 size={11} className="text-[var(--app-accent)]" />
        <span className="text-[length:var(--text-xs)] font-black text-black/30 uppercase tracking-wider">
          {t('laila.analyze.section.keyPoints')}
        </span>
      </div>
      <div className="space-y-1.5">
        {points.map((point, i) => (
          <div key={i} className="flex gap-2.5 items-start">
            <span className="text-[length:var(--text-xs)] font-black text-[var(--app-accent)] mt-0.5 shrink-0">
              {String(i + 1).padStart(2, '0')}
            </span>
            <p className="text-[length:var(--text-xs)] text-black/60 leading-[var(--leading-body)]">
              {point}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
