import { EmbeddingModelService } from '../../ai/EmbeddingModelService';
import { ConnectionManager } from '../../db/ConnectionManager';
import { EmbeddingService } from '../EmbeddingService';
import { getRagConfig } from '../config';
import { detectCapability } from '../search/capability';
import { searchFts } from '../search/fts';
import { mergeHybrid, type EmbeddingMode } from '../search/hybrid';
import { runVectorSearch } from '../search/runVectorSearch';
import { getCompatibleEmbeddingModels } from '../searchSupport';
import { embeddingCache, searchCache, buildSearchOptionsKey } from './cache';
import type { SearchResult, UnifiedSearchOptions } from './types';

interface UnifiedSearchLogger {
  info(message: string, meta?: unknown): void;
  warn(message: string, meta?: unknown): void;
  error(message: string, error?: unknown): void;
}

export async function performSimpleSearch(
  options: UnifiedSearchOptions,
  log: UnifiedSearchLogger,
): Promise<SearchResult[]> {
  const config = getRagConfig();
  const {
    query,
    topK = config.limits.topK,
    category,
    threshold = config.thresholds.cosineSimilarity,
  } = options;

  const db = ConnectionManager.get('kb_main');
  const model = await EmbeddingModelService.get();

  const textFallback = async (): Promise<SearchResult[]> => {
    log.warn('Falling back to FTS-only search');
    const textResults = await searchFts(
      db,
      query,
      topK * config.limits.vectorLimitMultiplier,
      category,
    );
    const fallback = textResults.slice(0, topK);

    // Fallback: dummy vector (all zeros) for semantic cache compatibility
    const dummyVector = new Float32Array(1024).fill(0);
    const optionsKey = buildSearchOptionsKey(model, topK, category, false, false, 0);
    searchCache.set(query, dummyVector, fallback, optionsKey);
    return fallback;
  };

  try {
    const [modelAvailable, compatibleModels] = await Promise.all([
      EmbeddingService.isModelAvailable(model),
      getCompatibleEmbeddingModels(db, model),
    ]);

    const hasCompatibleEmbeddings = compatibleModels.length > 0;
    log.info('[SimpleSearch] status:', {
      query: query.trim().slice(0, 40),
      model,
      modelAvailable,
      compatibleModels,
      hasCompatibleEmbeddings,
    });

    const embeddingMode: EmbeddingMode =
      modelAvailable && hasCompatibleEmbeddings ? 'real' : 'hash';
    if (!modelAvailable && !hasCompatibleEmbeddings) {
      log.warn('Ollama unavailable and no compatible embeddings -> FTS fallback');
      return textFallback();
    }

    const embeddingKey = `${query.trim()}:${model}`;
    let queryVector = embeddingCache.get(embeddingKey);
    if (!queryVector) {
      const { vector } = await EmbeddingService.embed(query, model, true);
      queryVector = vector;
      embeddingCache.set(embeddingKey, queryVector);
    }

    const capability = await detectCapability(db);
    const searchModels = hasCompatibleEmbeddings ? compatibleModels : [model];
    const vectorLimit = Math.min(topK * config.limits.vectorLimitMultiplier, 500);
    let vectorResults = await runVectorSearch(
      db,
      capability,
      queryVector,
      vectorLimit,
      category,
      searchModels,
    );

    log.info(`[SimpleSearch] Vector results: ${vectorResults.length}`);
    vectorResults = vectorResults.filter((result) => result.score >= threshold);
    log.info(`[SimpleSearch] After threshold: ${vectorResults.length} results`);

    if (!options.hybrid || query.trim().length < 3) {
      return vectorResults.slice(0, topK);
    }

    const textResults = await searchFts(db, query, vectorLimit, category);
    log.info(`[SimpleSearch] FTS results: ${textResults.length}`);

    if (vectorResults.length === 0 && textResults.length > 0) {
      log.warn('[SimpleSearch] No vector results, using FTS only');
      return textResults.slice(0, topK);
    }

    const finalResults = await mergeHybrid(vectorResults, textResults, topK, embeddingMode);
    log.info(`[SimpleSearch] Hybrid merged: ${finalResults.length}`);
    return finalResults;
  } catch (error) {
    log.error('[SimpleSearch] Error:', error);
    return textFallback();
  }
}
