import { normalizeTsvText } from '../../../rag/textExtractor/normalizeTsvText';
import type { RetrievalCandidate } from '../../../rag/hybridRetrievalService';

export function downweightFilter(candidates: RetrievalCandidate[]): RetrievalCandidate[] {
  return candidates
    .map((candidate) => {
      const importance = candidate.sourceImportance ?? 1;
      let penalty = 0;

      if ((candidate.vectorScore || 0) < 0.1 && (candidate.ftsScore || 0) < 0.05) {
        penalty += 0.2;
      }

      const adjustedScore = candidate.rrfScore * importance - penalty;
      if (importance < 1 || penalty > 0) {
        return { ...candidate, rrfScore: Math.max(0.005, adjustedScore) };
      }
      return candidate;
    })
    .sort((left, right) => right.rrfScore - left.rrfScore);
}

export function blendedFallback(candidates: RetrievalCandidate[], maxCandidates: number): RetrievalCandidate[] {
  return candidates
    .map((candidate) => {
      const fastScore = candidate.fastRerankScore ?? 0;
      const rrfScore = candidate.rrfScore ?? 0;
      return {
        ...candidate,
        llmRerankScore: fastScore * 0.6 + rrfScore * 0.4,
      };
    })
    .sort((left, right) => (right.llmRerankScore || 0) - (left.llmRerankScore || 0))
    .slice(0, maxCandidates);
}

export function mapSources(candidates: RetrievalCandidate[]) {
  return candidates.map((candidate, index) => ({
    index: index + 1,
    title: candidate.title,
    // FIX: нормализуем TSV/CSV перед показом в UI-карточке источника.
    // Без нормализации сырые табы/точки-с-запятыми видны в SemanticSourceCard.
    excerpt: normalizeTsvText(candidate.chunkText).substring(0, 1000),
    score: candidate.llmRerankScore || candidate.fastRerankScore || candidate.rrfScore,
    filePath: candidate.filePath,
    charStart: candidate.charStart,
    charEnd: candidate.charEnd,
    articleRef: candidate.articleRef,
  }));
}
