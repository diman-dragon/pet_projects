import { useEffect } from 'react';
import { Badge } from '../../../../components/ui';
import { UiText } from '../../../../components/ui/atoms/UiText';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { getAISummaryFallback } from '../../data';
import { useAIStream } from '../../hooks/useAIStream';
import { AIOfflineBanner } from './AIOfflineBanner';
import { AIHeader } from './AIHeader';
import { AITextBody } from './AITextBody';

interface AISummaryBlockProps {
  query: string;
  count: number;
}

export function AISummaryBlock({ query, count }: AISummaryBlockProps) {
  const { t } = useI18n();
  const { text, streaming, availability, start, reset, done } = useAIStream({
    agentId: 'knowledge',
  });

  useEffect(() => {
    if (!query.trim()) {
      reset();
      return;
    }

    if (count === 0) {
      reset();
      return;
    }

    const docWord = t('knowledge.ai.docWord', { count });
    const prompt = t('knowledge.ai.prompt', { query, count, docWord });

    start(prompt);
    return () => reset();
  }, [query, count, start, reset, t]);

  if (!query.trim()) return null;

  if (count === 0) {
    return (
      <div className="rounded-xl border border-black/[0.06] bg-white p-4">
        <AIHeader streaming={false} offline={true}>
          <Badge label={t('knowledge.ai.zeroBadge')} variant="gray" />
        </AIHeader>
        <UiText as="p" size="xs" tone="muted" className="leading-[var(--leading-body)]">
          {t('knowledge.ai.zeroBody')}
        </UiText>
      </div>
    );
  }

  const fallback = getAISummaryFallback(query, count);
  const resolvedText = text.trim() || (done || availability !== 'ready' ? fallback : '');
  const countLabel = t('knowledge.ai.resultsCount', { count });

  if (!streaming && !resolvedText) return null;

  return (
    <div className="rounded-xl border border-[var(--app-accent)]/15 bg-[var(--app-accent)]/[0.03] p-4 transition-all duration-300">
      <AIHeader streaming={streaming} offline={availability !== 'ready'}>
        <Badge label={countLabel} variant="blue" />
      </AIHeader>
      <AITextBody text={resolvedText} streaming={streaming && !resolvedText} />
      {availability !== 'ready' && (
        <div className="mt-2">
          <AIOfflineBanner timedOut={availability === 'timeout'} />
        </div>
      )}
    </div>
  );
}
