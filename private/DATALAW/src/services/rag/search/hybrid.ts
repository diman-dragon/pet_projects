// src/services/rag/search/hybrid.ts
//
// ИТЕРАЦИЯ 3: RRF k=60, language boost ×1.2, freshness penalty для старых virtual://.
//
// Изменения:
//   - RRF_K = 60 (явная константа, используется везде)
//   - Принимает queryLang для language boost
//   - Freshness penalty: virtual:// документы (клиенты/дела) получают дополнительный
//     downweight, если они появились ранее реальных документов в том же rank
//   - EmbeddingMode weights без изменений

import { createLogger } from '../../logger/Logger';
import { mergeHybridResults } from '../native/mergeHybridResults';
import type { SearchResult } from '../UnifiedSearchService';

const log = createLogger('hybrid');

export type EmbeddingMode = 'real' | 'hash';

const WEIGHTS: Record<EmbeddingMode, { vector: number; text: number }> = {
  real: { vector: 0.7, text: 0.3 },
  hash: { vector: 0.05, text: 0.95 },
};

export const RRF_K = 60;

/**
 * Language boost: если язык документа совпадает с языком запроса — усилить score.
 */
const LANG_BOOST = 1.2;

/**
 * Freshness penalty multiplier для virtual:// документов.
 * Клиент/дело без недавних обращений — менее релевантны для knowledge-запросов.
 */
const VIRTUAL_FRESHNESS_PENALTY = 0.85;

export async function mergeHybrid(
  vectorResults: SearchResult[],
  textResults: SearchResult[],
  topK: number,
  embeddingMode: EmbeddingMode = 'real',
  queryLang?: string,
  intent?: string,
): Promise<SearchResult[]> {
  // Базовые веса из embeddingMode
  let { vector: vectorWeight, text: textWeight } = WEIGHTS[embeddingMode];

  // Динамические веса по intent
  if (intent === 'legal') {
    // Точные номера статей лучше ищутся keyword-поиском
    textWeight = 1.2;
    vectorWeight = 0.8;
    log.debug('[mergeHybrid] Legal intent: boosting FTS weight to 1.2');
  } else if (intent === 'biographical') {
    // Семантический поиск лучше для биографических запросов
    textWeight = 0.7;
    vectorWeight = 1.3;
    log.debug('[mergeHybrid] Biographical intent: boosting vector weight to 1.3');
  }

  try {
    // Пробуем нативный Tauri merge (без language/freshness — применим после)
    const nativeResult = await mergeHybridResults(vectorResults, textResults, topK, vectorWeight, textWeight);
    return applyPostRankingAdjustments(nativeResult, queryLang);
  } catch (err) {
    log.warn('Tauri mergeHybridResults failed, using JS fallback:', err);
    return mergeHybridJS(vectorResults, textResults, topK, vectorWeight, textWeight, queryLang);
  }
}

/**
 * Применяет language boost и freshness penalty после merge.
 */
function applyPostRankingAdjustments(
  results: SearchResult[],
  queryLang?: string,
): SearchResult[] {
  if (!queryLang) return results;

  return results
    .map((r) => {
      let score = r.score;

      // Language boost
      if (r.lang && r.lang === queryLang) {
        score *= LANG_BOOST;
      }

      // Freshness penalty для virtual://
      if (r.filePath?.startsWith('virtual://')) {
        score *= VIRTUAL_FRESHNESS_PENALTY;
      }

      return { ...r, score: Math.min(score, 1) };
    })
    .sort((a, b) => b.score - a.score);
}

function mergeHybridJS(
  vectorResults: SearchResult[],
  textResults: SearchResult[],
  topK: number,
  vectorWeight: number,
  textWeight: number,
  queryLang?: string,
): SearchResult[] {
  const scoreMap = new Map<number, {
    result: SearchResult;
    vectorRank?: number;
    textRank?: number;
    vectorScore: number;
    textScore: number;
  }>();

  vectorResults.forEach((result, index) => {
    scoreMap.set(result.chunkId, {
      result,
      vectorRank: index + 1,
      vectorScore: result.score,
      textScore: 0,
    });
  });

  textResults.forEach((result, index) => {
    const existing = scoreMap.get(result.chunkId);
    if (existing) {
      existing.textRank = index + 1;
      existing.textScore = result.score;
    } else {
      scoreMap.set(result.chunkId, {
        result,
        textRank: index + 1,
        vectorScore: 0,
        textScore: result.score,
      });
    }
  });

  return Array.from(scoreMap.values())
    .map(({ result, vectorRank, textRank, vectorScore, textScore }) => {
      let score =
        (vectorRank ? vectorWeight * (1 / (RRF_K + vectorRank)) : 0) +
        (textRank ? textWeight * (1 / (RRF_K + textRank)) : 0);

      // Language boost
      if (queryLang && result.lang === queryLang) {
        score *= LANG_BOOST;
      }

      // Freshness penalty для virtual://
      if (result.filePath?.startsWith('virtual://')) {
        score *= VIRTUAL_FRESHNESS_PENALTY;
      }

      const matchType: SearchResult['matchType'] =
        vectorScore > 0 && textScore > 0 ? 'hybrid' : vectorScore > 0 ? 'vector' : 'text';

      return { ...result, score: Math.min(score, 1), matchType };
    })
    .sort((left, right) => right.score - left.score)
    .slice(0, topK);
}
