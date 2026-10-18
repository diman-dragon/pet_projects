// src/services/rag/utils/stemming.ts
//
// Unified stemming utilities for Russian and Ukrainian text.
// Used by fastReranker, FTS search, and GlobalSearchService.

/**
 * Common suffixes for Russian and Ukrainian declensions.
 * Covers the most frequent case endings to improve recall.
 */
export const STEM_SUFFIXES = [
  // Russian adjectives and possessives
  'ского', 'ской', 'ским', 'ском', 'ская', 'ские', 'ских',
  'ский', 'ього', 'ьою', 'ький',
  // Ukrainian adjectives
  'ькій', 'ська', 'ські', 'ського',
  // Russian/ Ukrainian genitive/dative endings
  'ого', 'ому',
  // Russian/ Ukrainian surname/name declensions (-ов/-ова/-ев/-ева)
  'ова', 'ову', 'овой', 'овым', 'овом',
  'ева', 'еву', 'евой', 'евым', 'евом',
] as const;

/**
 * Stems a token by removing known suffixes.
 * @param token - The input token to stem
 * @returns The stemmed token, or null if no stemming was applied (fts.ts behavior)
 */
export function stemToken(token: string): string | null {
  const lower = token.toLowerCase();
  for (const suffix of STEM_SUFFIXES) {
    if (lower.endsWith(suffix) && lower.length - suffix.length >= 4) {
      return lower.slice(0, lower.length - suffix.length);
    }
  }
  return null;
}

/**
 * Stems a token by removing known suffixes.
 * Returns the lowercased original if no stemming applied (fastReranker.ts behavior)
 * @param token - The input token to stem
 * @returns The stemmed token, or the lowercased original if no suffix matched
 */
export function stemTokenWithFallback(token: string): string {
  const lower = token.toLowerCase();
  for (const suffix of STEM_SUFFIXES) {
    if (lower.endsWith(suffix) && lower.length - suffix.length >= 4) {
      return lower.slice(0, lower.length - suffix.length);
    }
  }
  return lower;
}
