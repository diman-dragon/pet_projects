import { EmbeddingModelService } from '../ai/EmbeddingModelService';
import { EmbeddingService } from './EmbeddingService';
import { createLogger } from '../logger/Logger';
import { formatContext } from './search/formatContext';
import { countEmbeddingsForModel, countRows } from './searchSupport';
import { performAdvancedSearch } from './unifiedSearch/advanced';
import { buildSearchOptionsKey, embeddingCache, searchCache } from './unifiedSearch/cache';
import { performSimpleSearch } from './unifiedSearch/simple';
import type { SearchResult, UnifiedSearchOptions } from './unifiedSearch/types';
import { getRagConfig } from './config';

const log = createLogger('UnifiedSearchService');

export type { SearchResult, UnifiedSearchOptions } from './unifiedSearch/types';

export const UnifiedSearchService = {
  async search(options: UnifiedSearchOptions): Promise<SearchResult[]> {
    const config = getRagConfig();
    const {
      query,
      topK = config.limits.topK,
      category,
      hybrid = true,
      useReranker = hybrid,
      threshold = config.thresholds.cosineSimilarity,
    } = options;

    if (!query?.trim()) {
      return [];
    }

    const model = await EmbeddingModelService.get();

    // Семантический кеш требует эмбеддинг запроса
    const { vector: queryVector } = await EmbeddingService.embed(query, model, true);
    const optionsKey = buildSearchOptionsKey(model, topK, category, hybrid, useReranker, threshold);

    const cached = await searchCache.get(queryVector, optionsKey);
    if (cached) {
      log.debug('Semantic search cache hit', { query });
      return cached;
    }

    const result = hybrid
      ? await performAdvancedSearch({ ...options, queryVector })
      : await performSimpleSearch(options, log);

    searchCache.set(query, queryVector, result, optionsKey);
    return result;
  },

  formatContext,

  async hasIndex(): Promise<boolean> {
    try {
      return (await countRows('embeddings')) > 0;
    } catch {
      return false;
    }
  },

  async getStats(): Promise<{ documentsCount: number; chunksCount: number; embeddingsCount: number }> {
    const [documentsCount, chunksCount, embeddingsCount] = await Promise.all([
      countRows('library'),
      countRows('chunks'),
      countRows('embeddings'),
    ]);

    return { documentsCount, chunksCount, embeddingsCount };
  },

  clearCache(): void {
    searchCache.clear();
    embeddingCache.clear();
  },

  clearSearchCache(): void {
    searchCache.clear();
  },

  clearEmbeddingCache(): void {
    embeddingCache.clear();
  },

  getCacheStats() {
    return {
      search: searchCache.getStats(),
      embedding: embeddingCache.getStats(),
    };
  },

  async getEmbeddingCount(model: string): Promise<number> {
    try {
      return await countEmbeddingsForModel(model);
    } catch {
      return 0;
    }
  },
};
