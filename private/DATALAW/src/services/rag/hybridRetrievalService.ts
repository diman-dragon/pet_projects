// src/services/rag/hybridRetrievalService.ts
//
// FIX: queryLang из HybridRetrievalOptions теперь передаётся в performFtsRetrieval.
// До этого language-boost в FTS (×1.2) объявлен в runFtsQuery но никогда не применялся.

import { createLogger } from '../logger/Logger';
import { getRagConfig } from './config';
import { buildQueryList, buildRetrievalTrace } from './hybridRetrieval/helpers';
import { performRrfFusionWithDiversity } from './hybridRetrieval/fusion';
import { performFtsRetrieval, performVectorRetrieval } from './hybridRetrieval/search';
import type {
  HybridRetrievalOptions,
  RetrievalCandidate,
  RetrievalTrace,
  SimpleQueryResults,
} from './hybridRetrieval/types';

const log = createLogger('HybridRetrievalService');
const PER_QUERY_TOP_K = 20;

export type { HybridRetrievalOptions, RetrievalCandidate, RetrievalTrace } from './hybridRetrieval/types';
export async function performHybridRetrieval(
  options: HybridRetrievalOptions,
): Promise<RetrievalCandidate[]> {
  const config = getRagConfig();
  const {
    query,
    rewrittenQuery,
    expansions = [],
    intent,
    topK: topKInput,
    category,
    queryLang,
    queryVector, // ADDED
    onDebug,
  } = options;

  const topK = topKInput ?? config.limits.topK;
  const queries = buildQueryList(query, rewrittenQuery, expansions);

  log.debug('Starting hybrid retrieval', { query, queries: queries.length, topK, intent, queryLang });

  const searchResults: SimpleQueryResults[] = await Promise.all(
    queries.map(async (currentQuery, index) => {
      // HyDE: для основного запроса (index 0) используем гипотетический ответ для векторного поиска
      const vectorQuery = (index === 0 && options.hydeSnippet) ? options.hydeSnippet : currentQuery;

      // Если у нас уже есть вектор для основного запроса и мы НЕ используем HyDE — передаём его
      const existingVector = (index === 0 && !options.hydeSnippet) ? queryVector : undefined;

      return {
        query: currentQuery,
        vectorResults: await performVectorRetrieval(vectorQuery, PER_QUERY_TOP_K, category, log, existingVector),
        // FTS всегда идёт по оригинальному/переписанному запросу
        ftsResults: await performFtsRetrieval(currentQuery, PER_QUERY_TOP_K, category, log, queryLang),
      };
    }),
  );

  const candidates = performRrfFusionWithDiversity(searchResults, topK);
  const trace: RetrievalTrace = buildRetrievalTrace(queries, searchResults, candidates);

  onDebug?.(trace);

  log.debug('Hybrid retrieval completed', {
    queries: queries.length,
    candidates: candidates.length,
    uniqueFiles: new Set(candidates.map((candidate) => candidate.filePath)).size,
    vectorHits: searchResults.reduce((sum, item) => sum + item.vectorResults.length, 0),
    ftsHits: searchResults.reduce((sum, item) => sum + item.ftsResults.length, 0),
  });

  return candidates;
}
