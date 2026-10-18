// src/pages/laila/components/analyze/AnalyzeResult.tsx

import { UiText } from '../../../../components/ui/atoms/UiText';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { AnalysisResult } from '../../types';
import { AnalyzeConfidence } from './AnalyzeConfidence';
import { AnalyzeSummary } from './AnalyzeSummary';
import { AnalyzeKeyPoints } from './AnalyzeKeyPoints';
import { AnalyzeRisks } from './AnalyzeRisks';
import { AnalyzeArticles } from './AnalyzeArticles';

interface AnalyzeResultProps {
  result: AnalysisResult | null;
  isStreaming?: boolean;
  rawStreamingText?: string;
}

/**
 * Пытается вытащить данные из неполной JSON-строки.
 * Это позволяет компонентам рендериться до того, как закроется последняя скобка.
 */
function parsePartialJson(raw: string): Partial<AnalysisResult> {
  try {
    // Чистим текст от markdown оберток, если они есть в потоке
    const cleanText = raw
      .replace(/```json/g, '')
      .replace(/```/g, '')
      .trim();

    // Поиск полей с помощью регулярных выражений (самый надежный способ для стриминга)
    const extractString = (field: string) => {
      const match = cleanText.match(new RegExp(`"${field}"\\s*:\\s*"([^"]*)`));
      return match ? match[1] : '';
    };

    const extractArray = (field: string) => {
      const match = cleanText.match(new RegExp(`"${field}"\\s*:\\s*\\[([^\\]]*)`));
      if (!match) return [];
      return match[1]
        .split(',')
        .map((item) => item.replace(/"/g, '').trim())
        .filter((item) => item !== '');
    };

    const extractNumber = (field: string) => {
      const match = cleanText.match(new RegExp(`"${field}"\\s*:\\s*(\\d+)`));
      return match ? parseInt(match[1], 10) : 0;
    };

    return {
      summary: extractString('summary'),
      keyPoints: extractArray('keyPoints'),
      risks: extractArray('risks'),
      articles: extractArray('articles'),
      confidence: extractNumber('confidence'),
    };
  } catch (e) {
    return {};
  }
}

/** * Полный результат анализа.
 * Если идет стриминг — парсит текст на лету и показывает блоки сразу.
 */
export function AnalyzeResult({ result, isStreaming, rawStreamingText }: AnalyzeResultProps) {
  const { t } = useI18n();
  // Если есть готовый результат — берем его, иначе парсим текущий поток
  const data = result || parsePartialJson(rawStreamingText || '');

  // Проверяем наличие данных для отображения каждого блока
  const hasSummary = !!data.summary;
  const hasPoints = data.keyPoints && data.keyPoints.length > 0;
  const hasRisks = data.risks && data.risks.length > 0;
  const hasArticles = data.articles && data.articles.length > 0;
  const hasConfidence = (data.confidence ?? 0) > 0;

  return (
    <div className="select-text space-y-6 animate-in fade-in duration-500 pb-10">
      {/* 1. Уверенность (Confidence) */}
      {hasConfidence && (
        <div className="animate-in slide-in-from-top-2 duration-300">
          <AnalyzeConfidence value={data.confidence!} />
        </div>
      )}

      {/* 2. Краткое содержание */}
      {hasSummary && (
        <div className="animate-in fade-in slide-in-from-left-2 duration-500">
          <AnalyzeSummary text={data.summary!} />
        </div>
      )}

      {/* 3. Тезисы */}
      {hasPoints && (
        <div className="animate-in fade-in slide-in-from-left-2 duration-700">
          <AnalyzeKeyPoints points={data.keyPoints!} />
        </div>
      )}

      {/* 4. Риски */}
      {hasRisks && (
        <div className="animate-in fade-in slide-in-from-left-2 duration-700">
          <AnalyzeRisks risks={data.risks!} />
        </div>
      )}

      {/* 5. Статьи */}
      {hasArticles && (
        <div className="animate-in fade-in slide-in-from-bottom-2 duration-700">
          <AnalyzeArticles articles={data.articles!} />
        </div>
      )}

      {/* Индикатор печати, если мы еще в процессе */}
      {isStreaming && (
        <div className="flex items-center gap-2 px-1 opacity-40">
          <span className="w-1.5 h-1.5 bg-black rounded-full animate-bounce [animation-delay:-0.3s]"></span>
          <span className="w-1.5 h-1.5 bg-black rounded-full animate-bounce [animation-delay:-0.15s]"></span>
          <span className="w-1.5 h-1.5 bg-black rounded-full animate-bounce"></span>
          <UiText as="span" size="xs" weight="bold" className="uppercase tracking-widest ml-1">
            {t('laila.analyze.streaming')}
          </UiText>
        </div>
      )}
    </div>
  );
}
