// src/services/rag/hybridRetrieval/search.ts
//
// FIX: queryLang пробрасывается в performFtsRetrieval → searchFts → runFtsQuery.
// До этого language-boost (×1.2 для документов на языке запроса) в runFtsQuery
// был объявлен но никогда не получал значение — мёртвый код с нулевым эффектом.

import { EmbeddingModelService } from '../../ai/EmbeddingModelService';
import { ConnectionManager } from '../../db/ConnectionManager';
import { EmbeddingService } from '../EmbeddingService';
import { getRagConfig } from '../config';
import { detectCapability } from '../search/capability';
import { searchFts } from '../search/fts';
import { runVectorSearch } from '../search/runVectorSearch';
import type { RetrievalHit } from './types';

interface HybridRetrievalLogger {
  debug(message: string, meta?: unknown): void;
  error(message: string, error?: unknown): void;
}

export async function getCompatibleEmbeddingModels(db: any, model: string): Promise<string[]> {
  try {
    const exactRows = (await db.select(
      `SELECT COUNT(*) AS cnt FROM embeddings WHERE model = $1 LIMIT 1`,
      [model],
    )) as Array<{ cnt: number }> | undefined;

    if ((exactRows?.[0]?.cnt ?? 0) > 0) return [model];

    const modelBase = model.split(':')[0];
    if (modelBase && modelBase !== model) {
      const baseRows = (await db.select(
        `SELECT DISTINCT model FROM embeddings WHERE model LIKE $1 || '%' LIMIT 5`,
        [modelBase],
      )) as Array<{ model: string }> | undefined;

      const found = (baseRows ?? []).map((row) => row.model).filter(Boolean);
      if (found.length > 0) return found;
    }

    return [];
  } catch {
    return [];
  }
}
export async function performVectorRetrieval(
  query: string,
  topK: number,
  category: string | undefined,
  log: HybridRetrievalLogger,
  existingVector?: Float32Array,
): Promise<RetrievalHit[]> {
  try {
    const config = getRagConfig();
    const vectorMinScore = config.thresholds.vectorMin;

    const db = ConnectionManager.get('kb_main');
    const model = await EmbeddingModelService.get();

    const queryVector = existingVector || (await EmbeddingService.embed(query, model, true)).vector;

    const capability = await detectCapability(db);
    const compatibleModels = await getCompatibleEmbeddingModels(db, model);
    const searchModels = compatibleModels.length > 0 ? compatibleModels : [model];

    const results = await runVectorSearch(db, capability, queryVector, topK, category, searchModels);
    const filtered = results
      .filter((result) => result.score >= vectorMinScore)
      .map((result) => ({ chunkId: result.chunkId, score: result.score, data: result }));

    log.debug('Vector retrieval diagnostics', {
      query,
      rawCount: results.length,
      filteredCount: filtered.length,
      threshold: vectorMinScore,
    });

    return filtered;
  } catch (error) {
    log.error('Vector search failed', error);
    return [];
  }
}
// FIX: добавлен параметр queryLang — передаётся в searchFts → runFtsQuery
// для применения language-boost (×1.2 для документов на языке запроса).
export async function performFtsRetrieval(
  query: string,
  topK: number,
  category: string | undefined,
  log: HybridRetrievalLogger,
  queryLang?: string,
): Promise<RetrievalHit[]> {
  try {
    const config = getRagConfig();
    const ftsMinScore = config.thresholds.ftsMin;

    const db = ConnectionManager.get('kb_main');
    // FIX: передаём queryLang в searchFts
    const results = await searchFts(db, query, topK, category, [], queryLang);
    const filtered = results
      .filter((result) => result.score >= ftsMinScore)
      .map((result) => ({ chunkId: result.chunkId, score: result.score, data: result }));

    log.debug('FTS retrieval diagnostics', {
      query,
      rawCount: results.length,
      filteredCount: filtered.length,
      threshold: ftsMinScore,
      queryLang,
    });

    return filtered;
  } catch (error) {
    log.error('FTS search failed', error);
    return [];
  }
}
