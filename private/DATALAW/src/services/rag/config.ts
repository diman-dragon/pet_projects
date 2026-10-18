// src/services/rag/config.ts
//
// FIX v2:
// 1. vectorMin: 0.18 → 0.10 — BGE-M3 на русском юридическом тексте
//    выдаёт cosine similarity 0.08–0.30. Порог 0.18 отрезал большинство
//    релевантных чанков. Порог 0.10 соответствует реальному диапазону модели.
//
// 2. cosineSimilarity: 0.30 → 0.15 — аналогично, для JS cosine fallback.
//
// 3. chunkSize: 1200 → 800 символов для юридических текстов.
//    Каждая статья/пункт должны умещаться в один чанк.
//    BGE-M3 работает лучше на коротких фокусных текстах (≤200 токенов).
//    overlap: 150 → 100 (proportionally).
//    minChunkSize: 300 → 150.

export interface RagConfig {
  thresholds: {
    vectorMin: number;
    ftsMin: number;
    hardCutoff: number;
    cosineSimilarity: number;
  };
  weights: {
    cosine: number;
    keyword: number;
  };
  limits: {
    topK: number;
    maxCandidates: number;
    maxChunksPerSource: number;
    vectorLimitMultiplier: number;
    searchCacheTtlMs: number;
  };
  rrf: {
    k: number;
  };
  embedding: {
    maxRetries: number;
    timeoutMs: number;
  };
  chunking: {
    chunkSize: number;
    overlapSize: number;
    minChunkSize: number;
    maxChunkChars: number;
    childSize?: number;
  };
}

export const DEFAULT_RAG_CONFIG: RagConfig = {
  thresholds: {
    // FIX: снижен с 0.18 → 0.10
    // BGE-M3 на русском юридическом тексте: типичные scores 0.08–0.30
    // Порог 0.18 отфильтровывал большинство релевантных результатов
    vectorMin: 0.10,
    ftsMin: 0.10,
    hardCutoff: 0.08,
    // FIX: снижен с 0.30 → 0.15 для JS cosine fallback
    cosineSimilarity: 0.15,
  },
  weights: {
    cosine: 1.0,
    keyword: 1.0,
  },
  limits: {
    topK: 25,
    maxCandidates: 12,
    maxChunksPerSource: 5,
    vectorLimitMultiplier: 6,
    searchCacheTtlMs: 30_000,
  },
  rrf: {
    k: 60,
  },
  embedding: {
    maxRetries: 3,
    timeoutMs: 10_000,
  },
  chunking: {
    // Parent-Child strategy:
    // chunkSize (parent) ≈ 1500 characters
    // childSize ≈ 400 characters for retrieval
    chunkSize: 1500,
    overlapSize: 200,
    minChunkSize: 150,
    maxChunkChars: 2000,
    childSize: 400,
  },
};

let currentConfig = { ...DEFAULT_RAG_CONFIG };

export function getRagConfig(): RagConfig {
  return { ...currentConfig };
}

export function updateRagConfig(updates: Partial<RagConfig>): void {
  currentConfig = { ...currentConfig, ...updates };
}

export function resetRagConfig(): void {
  currentConfig = { ...DEFAULT_RAG_CONFIG };
}
