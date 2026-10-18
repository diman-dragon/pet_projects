// components/analyze/AnalyzeLoader.tsx

import { Sparkles } from 'lucide-react';
import { UiText } from '../../../../components/ui/atoms/UiText';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

/** Анимация загрузки на время анализа документа */
export function AnalyzeLoader() {
  const { t } = useI18n();

  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4">
      <div className="relative size-16">
        <div
          className="absolute inset-0 rounded-full bg-gradient-to-br from-violet-400/20 to-blue-500/20 animate-ping"
          style={{ animationDuration: '1.5s' }}
        />
        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-violet-500/80 to-[#0067c0]/80 flex items-center justify-center">
          <Sparkles size={20} className="text-white animate-pulse" />
        </div>
      </div>
      <div className="text-center">
        <UiText as="p" size="xs" weight="bold" tone="secondary">
          {t('laila.analyze.loaderTitle')}
        </UiText>
        <UiText as="p" size="xs" tone="muted" className="mt-1">
          {t('laila.analyze.loaderSubtitle')}
        </UiText>
      </div>
    </div>
  );
}
