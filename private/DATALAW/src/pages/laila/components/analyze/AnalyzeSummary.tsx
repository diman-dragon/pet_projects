// components/analyze/AnalyzeSummary.tsx

import { BookOpen } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

interface AnalyzeSummaryProps {
  text: string;
}

/** Блок краткого содержания документа */
export function AnalyzeSummary({ text }: AnalyzeSummaryProps) {
  const { t } = useI18n();

  return (
    <div>
      <div className="flex items-center gap-1.5 mb-2">
        <BookOpen size={11} className="text-black/30" />
        <span className="text-[length:var(--text-xs)] font-black text-black/30 uppercase tracking-wider">
          {t('laila.analyze.section.summary')}
        </span>
      </div>
      <p className="select-text cursor-text text-[length:var(--text-xs)] text-black/65 leading-[var(--leading-body)] bg-black/[0.02] rounded-xl p-3 border border-black/[0.05]">
        {text}
      </p>
    </div>
  );
}
