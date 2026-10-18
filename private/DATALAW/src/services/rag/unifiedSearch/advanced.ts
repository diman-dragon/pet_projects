import { getRagConfig } from '../config';
import type { SearchResult, UnifiedSearchOptions } from './types';

export async function performAdvancedSearch(options: UnifiedSearchOptions): Promise<SearchResult[]> {
  const { performHybridRetrieval } = await import('../hybridRetrievalService');
  const candidates = await performHybridRetrieval({
    query: options.query,
    rewrittenQuery: options.rewrittenQuery,
    expansions: options.expansions,
    intent: options.intent,
    hydeSnippet: options.hydeSnippet,
    topK: options.topK,
    category: options.category,
    onDebug: options.onDebug,
    queryVector: options.queryVector, // ADDED
  });

  const results: SearchResult[] = candidates.map((candidate) => ({
    chunkId: candidate.chunkId,
    libraryId: candidate.libraryId,
    title: candidate.title,
    category: candidate.category,
    filePath: candidate.filePath,
    chunkText: candidate.chunkText,
    chunkIndex: candidate.chunkIndex,
    score: candidate.rrfScore,
    matchType: candidate.matchType === 'fts' ? 'text' : candidate.matchType,
    charStart: candidate.charStart,
    charEnd: candidate.charEnd,
    articleRef: candidate.articleRef,
    parentId: candidate.parentId,
    parentText: candidate.parentText,
  }));

  if (!options.useReranker) {
    return results;
  }

  const { llmRerank } = await import('../llmReranker');
  const reranked = await llmRerank({
    query: options.query,
    candidates: candidates.map((candidate) => ({
      ...candidate,
      llmRerankScore: candidate.rrfScore,
    })),
    maxCandidates: getRagConfig().limits.maxCandidates,
  });

  const rerankMap = new Map(reranked.map((candidate) => [candidate.chunkId, candidate.llmRerankScore]));
  results.forEach((result) => {
    result.score = rerankMap.get(result.chunkId) ?? result.score;
  });
  results.sort((left, right) => right.score - left.score);

  return results;
}
