import { getRagConfig } from '../config';
import type { RetrievalCandidate, SimpleQueryResults } from './types';

function mapCandidate(chunkId: number, item: {
  data: any;
  rrfScore: number;
  vectorScore?: number;
  ftsScore?: number;
  matchType: RetrievalCandidate['matchType'];
}): RetrievalCandidate {
  return {
    chunkId,
    libraryId: Number(item.data.libraryId),
    title: String(item.data.title ?? ''),
    category: String(item.data.category ?? ''),
    filePath: String(item.data.filePath ?? ''),
    chunkText: String(item.data.chunkText ?? ''),
    chunkIndex: Number(item.data.chunkIndex ?? 0),
    charStart: item.data.charStart != null ? Number(item.data.charStart) : undefined,
    charEnd: item.data.charEnd != null ? Number(item.data.charEnd) : undefined,
    articleRef: item.data.articleRef != null ? String(item.data.articleRef) : null,
    vectorScore: item.vectorScore,
    ftsScore: item.ftsScore,
    rrfScore: item.rrfScore,
    matchType: item.matchType,
  };
}

function buildAllCandidates(searchResults: SimpleQueryResults[]): RetrievalCandidate[] {
  const config = getRagConfig();
  const rrfK = config.rrf.k;
  const grouped = new Map<number, {
    data: any;
    rrfScore: number;
    vectorScore?: number;
    ftsScore?: number;
    matchType: RetrievalCandidate['matchType'];
  }>();

  for (const result of searchResults) {
    for (let index = 0; index < result.vectorResults.length; index++) {
      const item = result.vectorResults[index];
      const contribution = 1.0 / (rrfK + index + 1);
      const existing = grouped.get(item.chunkId);

      if (existing) {
        existing.rrfScore += contribution;
        existing.vectorScore = Math.max(existing.vectorScore ?? 0, item.score);
        if (existing.ftsScore !== undefined) existing.matchType = 'hybrid';
      } else {
        grouped.set(item.chunkId, {
          data: item.data,
          rrfScore: contribution,
          vectorScore: item.score,
          matchType: 'vector',
        });
      }
    }

    for (let index = 0; index < result.ftsResults.length; index++) {
      const item = result.ftsResults[index];
      // FIX: FTS contribution поднят с 0.9 до 1.0 — равноправный с вектором.
      // Downweight систематически занижал точные keyword-попадания (ст. 450 ГК РФ).
      // При intent=legal keyword-поиск не менее важен чем семантический.
      const contribution = 1.0 / (rrfK + index + 1);
      const existing = grouped.get(item.chunkId);

      if (existing) {
        existing.rrfScore += contribution;
        existing.ftsScore = Math.max(existing.ftsScore ?? 0, item.score);
        existing.matchType = existing.vectorScore !== undefined ? 'hybrid' : 'fts';
      } else {
        grouped.set(item.chunkId, {
          data: item.data,
          rrfScore: contribution,
          ftsScore: item.score,
          matchType: 'fts',
        });
      }
    }
  }

  return Array.from(grouped.entries())
    .map(([chunkId, item]) => mapCandidate(chunkId, item))
    .filter((item) => item.rrfScore > 0)
    .sort((left, right) => right.rrfScore - left.rrfScore);
}

export function performRrfFusionWithDiversity(
  searchResults: SimpleQueryResults[],
  topK: number,
): RetrievalCandidate[] {
  const config = getRagConfig();
  const maxChunksPerSource = config.limits.maxChunksPerSource;
  const allCandidates = buildAllCandidates(searchResults);
  const fileChunkCount = new Map<string, number>();
  const diverseResult: RetrievalCandidate[] = [];

  for (const candidate of allCandidates) {
    const filePath = candidate.filePath || 'unknown';
    const count = fileChunkCount.get(filePath) ?? 0;
    if (count >= maxChunksPerSource) continue;

    fileChunkCount.set(filePath, count + 1);
    diverseResult.push(candidate);

    if (diverseResult.length >= topK) break;
  }

  if (diverseResult.length >= Math.min(5, topK)) {
    return diverseResult;
  }

  const softLimit = Math.min(8, topK);
  const alreadyAdded = new Set(diverseResult.map((candidate) => candidate.chunkId));
  const fileChunkCountSoft = new Map<string, number>(fileChunkCount);

  for (const candidate of allCandidates) {
    if (diverseResult.length >= softLimit) break;
    if (alreadyAdded.has(candidate.chunkId)) continue;

    const filePath = candidate.filePath || 'unknown';
    const count = fileChunkCountSoft.get(filePath) ?? 0;
    if (count >= Math.max(maxChunksPerSource, 8)) continue;

    fileChunkCountSoft.set(filePath, count + 1);
    diverseResult.push(candidate);
    alreadyAdded.add(candidate.chunkId);
  }

  return diverseResult;
}
