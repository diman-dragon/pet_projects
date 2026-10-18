import { CACHE_SIZE } from './constants';
import type { QueryUnderstandingResult } from './types';

const understandingCache = new Map<string, QueryUnderstandingResult>();

export function getCachedUnderstanding(cacheKey: string): QueryUnderstandingResult | null {
  return understandingCache.get(cacheKey) ?? null;
}

export function setCachedUnderstanding(cacheKey: string, value: QueryUnderstandingResult): void {
  understandingCache.set(cacheKey, value);
  if (understandingCache.size <= CACHE_SIZE) {
    return;
  }

  const firstKey = understandingCache.keys().next().value;
  if (firstKey) {
    understandingCache.delete(firstKey);
  }
}
