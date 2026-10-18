import { EmbeddingService } from './EmbeddingService';
import { IndexerService } from './IndexerService';
import { KnowledgeLibraryService } from './KnowledgeLibraryService';
import { UnifiedSearchService } from './UnifiedSearchService';
import { forceReembedAll as _forceReembedAll } from './indexer/forceReembedAll';
import { ClientIndexService, CLIENT_CATEGORY, CASE_CATEGORY } from '../clientSearch/ClientIndexService';
import type { UnifiedSearchOptions, SearchResult } from './UnifiedSearchService';
import type { LibraryDocumentWithStatus } from './knowledgeLibrary/types';
import type { ReembedProgress, ReembedResult } from './indexer/forceReembedAll';

export const RagFacade = {
  search(options: UnifiedSearchOptions): Promise<SearchResult[]> {
    return UnifiedSearchService.search(options);
  },
  searchClients(query: string, topK = 10): Promise<SearchResult[]> {
    return UnifiedSearchService.search({ query, topK, category: CLIENT_CATEGORY, hybrid: true });
  },
  searchCases(query: string, topK = 10): Promise<SearchResult[]> {
    return UnifiedSearchService.search({ query, topK, category: CASE_CATEGORY, hybrid: true });
  },
  async searchAll(query: string, topK = 15): Promise<SearchResult[]> {
    const [clients, cases, knowledge] = await Promise.all([
      UnifiedSearchService.search({ query, topK: 5, category: CLIENT_CATEGORY, hybrid: true }),
      UnifiedSearchService.search({ query, topK: 5, category: CASE_CATEGORY, hybrid: true }),
      UnifiedSearchService.search({ query, topK: 5, hybrid: true }),
    ]);
    return [...clients, ...cases, ...knowledge].sort((a, b) => b.score - a.score).slice(0, topK);
  },
  indexClient: ClientIndexService.indexClient.bind(ClientIndexService),
  indexCase: ClientIndexService.indexCase.bind(ClientIndexService),
  indexAllClients: ClientIndexService.indexAllClients.bind(ClientIndexService),
  removeClient: ClientIndexService.removeClient.bind(ClientIndexService),
  removeCase: ClientIndexService.removeCase.bind(ClientIndexService),
  getStats() { return UnifiedSearchService.getStats(); },
  hasIndex() { return UnifiedSearchService.hasIndex(); },
  getEmbeddingCount(model: string) { return UnifiedSearchService.getEmbeddingCount(model); },
  getLibraryDocuments(): Promise<LibraryDocumentWithStatus[]> { return KnowledgeLibraryService.getAll(); },
  updateLibraryDocument(id: number, data: { title?: string; category?: string; metadata?: Record<string, unknown> }) {
    UnifiedSearchService.clearSearchCache(); // Clear search cache on document updates
    return KnowledgeLibraryService.update(id, data);
  },
  deleteLibraryDocument(id: number) {
    UnifiedSearchService.clearSearchCache(); // Clear search cache on document deletion
    return KnowledgeLibraryService.delete(id);
  },
  isEmbeddingModelAvailable(model: string) { return EmbeddingService.isModelAvailable(model); },
  indexFolder: IndexerService.indexFolder,
  indexFile: IndexerService.indexFile,
  forceReembedAll(onProgress?: (p: ReembedProgress) => void, signal?: AbortSignal): Promise<ReembedResult> {
    return _forceReembedAll(onProgress, signal);
  },

  // Cache management methods
  clearAllCaches(): void {
    UnifiedSearchService.clearCache();
    EmbeddingService.clearCache();
  },

  clearSearchCache(): void {
    UnifiedSearchService.clearSearchCache();
  },

  clearEmbeddingCache(): void {
    UnifiedSearchService.clearEmbeddingCache();
  },

  getCacheMetrics() {
    return UnifiedSearchService.getCacheStats();
  },
};
