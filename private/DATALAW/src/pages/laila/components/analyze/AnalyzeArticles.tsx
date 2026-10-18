// components/analyze/AnalyzeArticles.tsx

import { Badge } from '../../../../components/ui';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

interface AnalyzeArticlesProps {
  articles: string[];
}

/** Применимые нормы в виде бейджей */
export function AnalyzeArticles({ articles }: AnalyzeArticlesProps) {
  const { t } = useI18n();

  return (
    <div>
      <div className="flex items-center gap-1.5 mb-2">
        <span className="text-[length:var(--text-xs)] font-black text-black/30 uppercase tracking-wider">
          {t('laila.analyze.section.articles')}
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {articles.map((a) => (
          <Badge key={a} label={a} variant="violet" />
        ))}
      </div>
    </div>
  );
}
