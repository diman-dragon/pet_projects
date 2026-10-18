import { LRUCache } from '../cache';
import type { SearchResult } from './types';
import { cosineSimilarity } from '../native/cosineSimilarity';

export const embeddingCache = new LRUCache<Float32Array>(200, 300_000);

interface SemanticEntry {
  query: string;
  vector: Float32Array;
  results: SearchResult[];
  optionsKey: string;
  expiresAt: number;
}

export class SemanticSearchCache {
  private entries: SemanticEntry[] = [];

  constructor(
    private readonly maxSize = 100,
    private readonly defaultTTL = 60_000,
    private readonly similarityThreshold = 0.95,
  ) {}

  async get(vector: Float32Array, optionsKey: string): Promise<SearchResult[] | undefined> {
    const now = Date.now();
    this.entries = this.entries.filter(e => e.expiresAt > now);

    for (const entry of this.entries) {
      if (entry.optionsKey !== optionsKey) continue;

      const sim = await cosineSimilarity(vector, entry.vector);
      if (sim >= this.similarityThreshold) {
        return entry.results;
      }
    }
    return undefined;
  }

  set(query: string, vector: Float32Array, results: SearchResult[], optionsKey: string, ttl?: number): void {
    if (this.entries.length >= this.maxSize) {
      this.entries.shift();
    }

    this.entries.push({
      query,
      vector,
      results,
      optionsKey,
      expiresAt: Date.now() + (ttl ?? this.defaultTTL),
    });
  }

  clear(): void {
    this.entries = [];
  }

  getStats() {
    return {
      size: this.entries.length,
      maxSize: this.maxSize,
    };
  }
}

export const searchCache = new SemanticSearchCache(100, 60_000, 0.92);

export function buildSearchOptionsKey(
  model: string,
  topK: number,
  category: string | undefined,
  hybrid: boolean,
  useReranker: boolean,
  threshold: number,
): string {
  return JSON.stringify({
    model,
    topK,
    category: category ?? '',
    hybrid,
    useReranker,
    threshold,
  });
}

export function buildSearchCacheKey(
  query: string,
  model: string,
  topK: number,
  category: string | undefined,
  hybrid: boolean,
  useReranker: boolean,
  threshold: number,
): string {
  return JSON.stringify({
    query: query.trim(),
    model,
    topK,
    category: category ?? '',
    hybrid,
    useReranker,
    threshold,
  });
}
