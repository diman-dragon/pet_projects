import type { RetrievalHit, RetrievalTrace, SimpleQueryResults, TraceResult } from './types';
import type { RetrievalCandidate } from './types';

const MAX_QUERIES = 3;

export function buildQueryList(original: string, rewritten?: string, expansions: string[] = []): string[] {
  const queries = new Set<string>();
  const normalizedOriginal = original.trim();
  const normalizedRewritten = rewritten?.trim();

  if (normalizedOriginal) queries.add(normalizedOriginal);
  if (normalizedRewritten && normalizedRewritten !== normalizedOriginal) {
    queries.add(normalizedRewritten);
  }
  for (const expansion of expansions.slice(0, 2)) {
    if (expansion && expansion.trim()) {
      queries.add(expansion.trim());
    }
  }

  return Array.from(queries).slice(0, MAX_QUERIES);
}

export function toTraceResult(item: RetrievalHit): TraceResult {
  return {
    chunkId: item.chunkId,
    title: String(item.data?.title ?? ''),
    score: Number(item.score ?? 0),
    category: String(item.data?.category ?? ''),
    filePath: String(item.data?.filePath ?? ''),
  };
}

export function buildRetrievalTrace(
  queries: string[],
  searchResults: SimpleQueryResults[],
  candidates: RetrievalCandidate[],
): RetrievalTrace {
  return {
    plannedQueries: queries.map((query) => ({ query, kind: 'search', weight: 1.0 })),
    vectorResults: searchResults.map((item) => ({
      query: item.query,
      kind: 'vector',
      results: item.vectorResults.slice(0, 10).map(toTraceResult),
    })),
    ftsResults: searchResults.map((item) => ({
      query: item.query,
      kind: 'fts',
      results: item.ftsResults.slice(0, 10).map(toTraceResult),
    })),
    mergedResults: candidates.slice(0, 20).map((candidate) => ({
      chunkId: candidate.chunkId,
      title: candidate.title,
      category: candidate.category,
      filePath: candidate.filePath,
      rrfScore: candidate.rrfScore,
      matchType: candidate.matchType,
      vectorScore: candidate.vectorScore,
      ftsScore: candidate.ftsScore,
    })),
  };
}
