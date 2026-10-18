import { createLogger } from '../logger/Logger';
import { fetchEmbedding, isModelAvailable, resetAvailabilityCache } from './embedding/local';
import { serialize, deserialize } from './embedding/serialize';

export interface EmbeddingResult {
  vector: Float32Array;
  dim: number;
}

const AVAILABILITY_TTL_MS = 5 * 60 * 1_000;

/**
 * Pure JS L2 normalization — no Tauri IPC dependency.
 * Vectors from Ollama are usually already normalized, but we normalize
 * again to be safe (e.g. after hash fallback).
 */
function normalizeVectorJS(vector: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vector.length; i++) {
    norm += vector[i] * vector[i];
  }
  norm = Math.sqrt(norm);
  if (norm === 0) return vector;
  const result = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i++) {
    result[i] = vector[i] / norm;
  }
  return result;
}

/**
 * Pure JS dot product cosine similarity.
 * Requires pre-normalized vectors (norm=1).
 */
function cosineSimilarityJS(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return Math.min(Math.max(sum, -1), 1);
}

class EmbeddingServiceClass {
  private readonly cache = new Map<string, EmbeddingResult>();
  private readonly log = createLogger('EmbeddingService');
  private availabilityCache = new Map<string, { available: boolean; expiresAt: number }>();

  async embed(text: string, model: string, isQuery: boolean = true): Promise<EmbeddingResult> {
    const normalized = text.trim();
    const cacheKey = `${model}:${isQuery ? 'q' : 'd'}:${normalized}`;
    const cached = this.cache.get(cacheKey);

    if (cached) {
      // LRU: move to end (most recently used)
      this.cache.delete(cacheKey);
      this.cache.set(cacheKey, cached);
      return { vector: new Float32Array(cached.vector), dim: cached.dim };
    }

    const raw = await fetchEmbedding(text, model, isQuery);
    // Pure JS normalization — no Tauri IPC
    const vector = normalizeVectorJS(raw);
    const result = { vector, dim: vector.length };
    this.cache.set(cacheKey, result);

    // LRU eviction: remove oldest (first) entries when cache grows large
    if (this.cache.size > 200) {
      const evictCount = 20;
      const keys = Array.from(this.cache.keys());
      for (let i = 0; i < evictCount && i < keys.length; i++) {
        this.cache.delete(keys[i]);
      }
    }

    return { vector: new Float32Array(vector), dim: result.dim };
  }

  serialize(vector: Float32Array): number[] {
    return Array.from(vector);
  }

  deserialize(blob: unknown): Float32Array {
    return deserialize(blob);
  }

  /** Pure JS cosine similarity — no Tauri dependency */
  cosineSimilarity(a: Float32Array, b: Float32Array): Promise<number> {
    return Promise.resolve(cosineSimilarityJS(a, b));
  }

  async isModelAvailable(model: string): Promise<boolean> {
    const now = Date.now();
    const entry = this.availabilityCache.get(model);
    if (entry && entry.expiresAt > now) {
      return entry.available;
    }

    const available = await isModelAvailable(model);
    this.availabilityCache.set(model, { available, expiresAt: now + AVAILABILITY_TTL_MS });
    return available;
  }

  clearCache(): void {
    this.log.info('Embedding cache cleared');
    this.cache.clear();
    this.availabilityCache.clear();
    resetAvailabilityCache();
  }
}

export const EmbeddingService = new EmbeddingServiceClass();

export { serialize as serializeVector, deserialize as deserializeVector };
