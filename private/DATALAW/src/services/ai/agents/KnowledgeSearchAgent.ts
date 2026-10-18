import { createLogger } from '../../logger/Logger';
import { searchLogger } from '../../logger/SearchLogger';
import type { SearchResult } from '../../rag/UnifiedSearchService';
import { understandQuery } from '../../rag/queryUnderstanding';
import { performHybridRetrieval, type RetrievalCandidate } from '../../rag/hybridRetrievalService';
import { fastRerank } from '../../rag/fastReranker';
import { llmRerank } from '../../rag/llmReranker';
import { generateGroundedAnswer, handleConversationalQuery, buildFallbackAnswer } from './knowledgeSearch/groundedAnswer';
import { legacyAnalyze } from './knowledgeSearch/legacyAnalyze';
import { streamText } from './knowledgeSearch/pipelineShared';
import { blendedFallback, downweightFilter, mapSources } from './knowledgeSearch/ranking';
import type { KnowledgeSearchResponse } from './knowledgeSearch/sourceTypes';
import type { HistoryMessage } from './knowledgeSearch/types';

const log = createLogger('KnowledgeSearchAgent');

interface SessionMetrics {
  query: string;
  retrievalCount: number;
  filteredCount: number;
  rerankedCount: number;
  finalChunks: number;
}

function createSessionMetrics(): SessionMetrics {
  return {
    query: '',
    retrievalCount: 0,
    filteredCount: 0,
    rerankedCount: 0,
    finalChunks: 0,
  };
}

export const KnowledgeSearchAgent = {
  async analyze(
    query: string,
    results: SearchResult[],
    onToken: (token: string) => void,
    signal?: AbortSignal,
    queryLang?: string,
  ): Promise<string> {
    if (results.length > 0) {
      return legacyAnalyze(query, results, onToken, signal, queryLang);
    }

    const result = await productionPipeline(query, onToken, signal, undefined, queryLang);
    return result.answer;
  },

  async search(
    query: string,
    onToken: (token: string) => void,
    signal?: AbortSignal,
    chatId?: string,
    queryLang?: string,
    history: HistoryMessage[] = [],
  ): Promise<KnowledgeSearchResponse> {
    return productionPipeline(query, onToken, signal, chatId, queryLang, history);
  },
};

async function productionPipeline(
  query: string,
  onToken: (token: string) => void,
  signal?: AbortSignal,
  chatId?: string,
  queryLang?: string,
  history: HistoryMessage[] = [],
): Promise<KnowledgeSearchResponse> {
  const startTime = Date.now();
  const sessionMetrics = createSessionMetrics();

  try {
    log.info('[RAG] Starting strict pipeline', { query, chatId, queryLang });

    const understanding = await understandQuery(query);
    const effectiveQueryLang = queryLang ?? understanding.lang ?? 'ru';
    sessionMetrics.query = understanding.rewritten_query;

    await logPipelineStart(query, understanding, effectiveQueryLang);

    if (understanding.intent === 'conversational') {
      const conversationalAnswer = await handleConversationalQuery(query, onToken, signal, queryLang, history);
      return { answer: conversationalAnswer, sources: [] };
    }

    const retrievalCandidates = await retrieveCandidates(query, understanding, sessionMetrics, effectiveQueryLang);
    if (retrievalCandidates.length === 0) {
      const noResults = 'На основе доступных документов не найдена информация по этому запросу.';
      streamText(noResults, onToken, signal);
      return { answer: noResults, sources: [] };
    }

    const { finalCandidates, usedFallback } = await rerankCandidates(understanding.rewritten_query, retrievalCandidates, sessionMetrics);
    const context = await buildContext(finalCandidates, understanding.rewritten_query, sessionMetrics);
    const answer = await produceAnswer(understanding.rewritten_query, context.chunks, onToken, signal, queryLang, history);

    await logPipelineFinish(query, effectiveQueryLang, startTime, sessionMetrics, answer, usedFallback);
    return { answer, sources: mapSources(context.chunks) };
  } catch (error) {
    log.error('[RAG] Pipeline failed', error);
    await searchLogger.logError('PIPELINE_ERROR', error, { query, chatId, queryLang });
    const errorMessage = 'Произошла ошибка при обработке запроса.';
    streamText(errorMessage, onToken, signal);
    return { answer: errorMessage, sources: [] };
  }
}

async function logPipelineStart(query: string, understanding: Awaited<ReturnType<typeof understandQuery>>, effectiveQueryLang: string): Promise<void> {
  await searchLogger.logQuery(query, {
    memory: 0,
    queryLang: effectiveQueryLang,
    queryType: understanding.intent,
  });

  await searchLogger.log('STAGE_1_PREPROCESSING', {
    originalQuery: query,
    normalized: query.toLowerCase().trim().replace(/\s+/g, ' '),
    intentHint: 'classified by regex patterns',
    queryLang: effectiveQueryLang,
  });

  await searchLogger.log('STAGE_1_QUERY_UNDERSTANDING', {
    rewritten: understanding.rewritten_query,
    expansions: understanding.expansions,
    intent: understanding.intent,
    expansionCount: understanding.expansions.length,
    queryLang: effectiveQueryLang,
  });
}

async function retrieveCandidates(
  query: string,
  understanding: Awaited<ReturnType<typeof understandQuery>>,
  sessionMetrics: SessionMetrics,
  queryLang?: string,
): Promise<RetrievalCandidate[]> {
  const retrievalStart = Date.now();
  const retrievalCandidates = await performHybridRetrieval({
    query,
    rewrittenQuery: understanding.rewritten_query,
    expansions: understanding.expansions.slice(0, 2),
    intent: understanding.intent,
    hydeSnippet: understanding.hyde_snippet,
    // FIX: передаём queryLang — активирует language-boost (×1.2) в FTS
    queryLang,
    topK: 30,
  });

  sessionMetrics.retrievalCount = retrievalCandidates.length;
  await searchLogger.log('STAGE_2_RETRIEVAL', {
    candidatesFound: retrievalCandidates.length,
    timing: `${Date.now() - retrievalStart}ms`,
  });

  return retrievalCandidates;
}

async function rerankCandidates(
  rewrittenQuery: string,
  retrievalCandidates: RetrievalCandidate[],
  sessionMetrics: SessionMetrics,
): Promise<{ finalCandidates: RetrievalCandidate[]; usedFallback: boolean }> {
  const filtered = downweightFilter(retrievalCandidates);
  sessionMetrics.filteredCount = filtered.length;

  const fastRerankStart = Date.now();
  const fastReranked = await fastRerank({
    query: rewrittenQuery,
    candidates: filtered,
    maxCandidates: 20,
  });

  await searchLogger.log('STAGE_4_FAST_RERANK', {
    inputCount: filtered.length,
    outputCount: fastReranked.length,
    timing: `${Date.now() - fastRerankStart}ms`,
  });

  let usedFallback = false;
  let finalCandidates: RetrievalCandidate[];
  const llmRerankStart = Date.now();

  try {
    finalCandidates = await llmRerank({
      query: rewrittenQuery,
      candidates: fastReranked,
      maxCandidates: 12,
    });

    if (finalCandidates.length === 0) {
      return { finalCandidates: [], usedFallback };
    }
  } catch (error) {
    log.warn('[RAG] LLM rerank failed, using blended fallback', error);
    usedFallback = true;
    finalCandidates = blendedFallback(fastReranked, 12);
  }

  sessionMetrics.rerankedCount = finalCandidates.length;
  await searchLogger.logLLMRerank(
    { count: fastReranked.length },
    {
      count: finalCandidates.length,
      topScore: finalCandidates[0]?.llmRerankScore ?? finalCandidates[0]?.fastRerankScore,
    },
    undefined,
    { duration: Date.now() - llmRerankStart },
  );

  return { finalCandidates, usedFallback };
}

async function buildContext(
  finalCandidates: RetrievalCandidate[],
  rewrittenQuery: string,
  sessionMetrics: SessionMetrics,
): Promise<{ chunks: RetrievalCandidate[] }> {
  if (finalCandidates.length === 0) {
    return { chunks: [] };
  }

  const { extractContext } = await import('../../rag/extractiveContext');
  const context = extractContext(finalCandidates, rewrittenQuery);
  sessionMetrics.finalChunks = context.chunks.length;
  return context;
}

async function produceAnswer(
  rewrittenQuery: string,
  chunks: RetrievalCandidate[],
  onToken: (token: string) => void,
  signal: AbortSignal | undefined,
  queryLang: string | undefined,
  history: HistoryMessage[],
): Promise<string> {
  if (chunks.length === 0) {
    const noInfoMessage = 'В документах нет информации по этому вопросу.';
    streamText(noInfoMessage, onToken, signal);
    return noInfoMessage;
  }

  try {
    return await generateGroundedAnswer(
      rewrittenQuery,
      chunks,
      onToken,
      signal,
      queryLang,
      history,
    );
  } catch (error) {
    // FIX: buildFallbackAnswer теперь async — awaiting
    const fallback = await buildFallbackAnswer(error, chunks.map((chunk) => chunk.title));
    streamText(fallback, onToken, signal);
    return fallback;
  }
}

async function logPipelineFinish(
  query: string,
  effectiveQueryLang: string,
  startTime: number,
  sessionMetrics: SessionMetrics,
  answer: string,
  usedFallback: boolean,
): Promise<void> {
  const totalTime = Date.now() - startTime;

  await searchLogger.log('STAGE_7_ANSWER_GENERATION', {
    query: sessionMetrics.query,
    answerLength: answer.length,
    timing: `${totalTime}ms`,
    queryLang: effectiveQueryLang,
  });

  await searchLogger.log('PIPELINE_COMPLETE', {
    originalQuery: query,
    totalTimeMs: totalTime,
    metrics: sessionMetrics,
    queryLang: effectiveQueryLang,
    usedFallback,
  });
}
