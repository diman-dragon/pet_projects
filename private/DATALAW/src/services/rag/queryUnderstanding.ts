import { createLogger } from '../logger/Logger';
import { detectLang } from '../search/languageDetection';
import { fastUnderstand, preprocessQuery } from './queryUnderstanding/helpers';
import { getCachedUnderstanding, setCachedUnderstanding } from './queryUnderstanding/cache';
import { requestRemoteUnderstanding } from './queryUnderstanding/remote';
import type { QueryUnderstandingResult } from './queryUnderstanding/types';

const log = createLogger('QueryUnderstanding');

export type { QueryUnderstandingResult } from './queryUnderstanding/types';

export async function understandQuery(query: string): Promise<QueryUnderstandingResult> {
  const cacheKey = query.trim().toLowerCase();
  const cached = getCachedUnderstanding(cacheKey);
  if (cached) {
    log.debug('Query understanding cache hit', { query });
    return cached;
  }

  const preprocessed = preprocessQuery(query);
  const isSimpleQuery = query.trim().split(/\s+/).length <= 5;

  if (isSimpleQuery) {
    const fastResult = withDetectedLanguage(fastUnderstand(query, preprocessed.intentHint), query);
    setCachedUnderstanding(cacheKey, fastResult);
    log.debug('Query understanding (fast rule-based)', { query, ...fastResult });
    return fastResult;
  }

  try {
    const remoteResult = await requestRemoteUnderstanding(query, preprocessed);
    setCachedUnderstanding(cacheKey, remoteResult);
    log.debug('Query understanding (LLM)', {
      query,
      rewritten: remoteResult.rewritten_query,
      expansions: remoteResult.expansions,
      intent: remoteResult.intent,
    });
    return remoteResult;
  } catch (error) {
    log.warn('LLM understanding failed, using fast fallback', error);
    const fallback = withDetectedLanguage(fastUnderstand(query, preprocessed.intentHint), query);
    setCachedUnderstanding(cacheKey, fallback);
    return fallback;
  }
}

function withDetectedLanguage(result: QueryUnderstandingResult, query: string): QueryUnderstandingResult {
  return {
    ...result,
    lang: detectLang(query, 'ru') ?? 'ru',
  };
}
