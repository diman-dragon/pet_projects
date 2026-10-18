// components/analyze/AnalyzeEmpty.tsx

import { Sparkles } from 'lucide-react';
import { UiText } from '../../../../components/ui/atoms/UiText';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

/** Заглушка до первого запуска анализа */
export function AnalyzeEmpty() {
  const { t } = useI18n();

  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center">
      <div className="size-14 rounded-2xl bg-violet-50 flex items-center justify-center">
        <Sparkles size={22} className="text-violet-400" />
      </div>
      <UiText as="p" size="xs" weight="bold" tone="muted">
        {t('laila.analyze.emptyTitle')}
      </UiText>
      <UiText as="p" size="xs" tone="muted" className="max-w-[220px] leading-[var(--leading-body)]">
        {t('laila.analyze.emptyBody')}
      </UiText>
    </div>
  );
}
