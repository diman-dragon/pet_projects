// src/services/rag/embedding/local.ts
//
// FIX v2:
// 1. checkOllamaModelAvailable — нормализованное сравнение имён моделей.
//    'bge-m3:latest' и 'bge-m3' считаются одной моделью.
//    Без этого Ollama отдаёт 'bge-m3' в /api/tags, мы ищем 'bge-m3:latest'
//    → available=false → hash fallback → семантика выключена.
//
// 2. warnHashFallbackOnce → console.error (не warn).
//    Hash fallback — критическая ситуация: hash(query) vs bge-m3(doc) = cosine≈0
//    → 0 векторных результатов → только FTS.

import { getEmbeddingModelDefinition } from '../../ai/embeddingModels';

const OLLAMA_API_URL = 'http://127.0.0.1:11434';
const OLLAMA_TIMEOUT_MS = 30_000;

let hashFallbackWarned = false;
let availabilityCache: { model: string; available: boolean; expiresAt: number } | null = null;

function warnHashFallbackOnce(reason: string): void {
  if (hashFallbackWarned) return;
  hashFallbackWarned = true;
  console.error(
    `[EmbeddingService] HASH FALLBACK ACTIVE — семантический поиск ОТКЛЮЧЁН.\n` +
    `Причина: ${reason}\n` +
    `Поиск работает только через ключевые слова (FTS).\n` +
    `Исправление: ollama serve && ollama pull bge-m3:latest`,
  );
  try {
    window.dispatchEvent(new CustomEvent('embedding:degraded', {
      detail: { reason, severity: 'critical' },
    }));
  } catch { /* non-browser */ }
}

function applyEmbeddingTaskPrefix(text: string, model: string, isQuery: boolean): string {
  const trimmed = text.trim();
  if (!trimmed) return trimmed;
  const def = getEmbeddingModelDefinition(model);
  switch (def.promptStyle) {
    case 'nomic': {
      const qp = 'search_query: ';
      const dp = 'search_document: ';
      if (trimmed.startsWith(qp) || trimmed.startsWith(dp)) return trimmed;
      return (isQuery ? qp : dp) + trimmed;
    }
    case 'mixedbread': {
      const p = 'Represent this sentence for searching relevant passages: ';
      return isQuery && !trimmed.startsWith(p) ? p + trimmed : trimmed;
    }
    case 'bge':
    case 'none':
    default:
      return trimmed;
  }
}

async function fetchOllamaEmbedding(text: string, model: string): Promise<Float32Array> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), OLLAMA_TIMEOUT_MS);
  try {
    const response = await fetch(`${OLLAMA_API_URL}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, prompt: text }),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`Ollama embeddings HTTP ${response.status}: ${response.statusText}`);
    }
    const json = await response.json();
    const embedding: number[] = json.embedding;
    if (!Array.isArray(embedding) || embedding.length === 0) {
      throw new Error('Ollama returned empty embedding array');
    }
    return new Float32Array(embedding);
  } finally {
    clearTimeout(timer);
  }
}

function hashString(value: string): number {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function hashEmbed(text: string, model: string, isQuery: boolean): Float32Array {
  const def = getEmbeddingModelDefinition(model);
  const dim = def.dim;
  const normalized = `${isQuery ? 'q' : 'd'}:${text.trim().toLowerCase()}`;
  const vector = new Float32Array(dim);
  const grams = normalized.length < 3
    ? [normalized]
    : Array.from({ length: normalized.length - 2 }, (_, i) => normalized.slice(i, i + 3));
  for (const gram of grams) {
    const hash = hashString(gram);
    const primary = hash % dim;
    const secondary = (hash >>> 8) % dim;
    const sign = (hash & 1) === 0 ? 1 : -1;
    vector[primary] += sign * 1.0;
    vector[secondary] += sign * 0.35;
  }
  return vector;
}

// FIX: нормализация — убираем :latest для сравнения
function normalizeModelId(m: string): string {
  return m.replace(/:latest$/i, '').trim().toLowerCase();
}

async function checkOllamaModelAvailable(model: string): Promise<boolean> {
  const now = Date.now();
  if (availabilityCache && availabilityCache.model === model && availabilityCache.expiresAt > now) {
    return availabilityCache.available;
  }
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 3000);
    const response = await fetch(`${OLLAMA_API_URL}/api/tags`, { signal: controller.signal });
    clearTimeout(timer);
    if (!response.ok) {
      availabilityCache = { model, available: false, expiresAt: now + 60_000 };
      return false;
    }
    const json = await response.json();
    const models: { name: string }[] = json.models ?? [];
    const modelNorm = normalizeModelId(model);
    // FIX: сравниваем нормализованные имена (без :latest)
    const available = models.some((item) => normalizeModelId(item.name) === modelNorm);
    availabilityCache = { model, available, expiresAt: now + 60_000 };
    return available;
  } catch {
    availabilityCache = { model, available: false, expiresAt: now + 10_000 };
    return false;
  }
}

export async function fetchEmbedding(
  text: string,
  model: string,
  isQuery: boolean,
): Promise<Float32Array> {
  const preparedText = applyEmbeddingTaskPrefix(text, model, isQuery);
  if (model === 'localmind-hash-embed-v1') return hashEmbed(preparedText, model, isQuery);
  const ollamaAvailable = await checkOllamaModelAvailable(model);
  if (!ollamaAvailable) {
    warnHashFallbackOnce(
      `Модель "${model}" не найдена или Ollama не запущена. Выполните: ollama pull ${model}`,
    );
    return hashEmbed(preparedText, model, isQuery);
  }
  try {
    const result = await fetchOllamaEmbedding(preparedText, model);
    hashFallbackWarned = false;
    return result;
  } catch (error) {
    warnHashFallbackOnce(
      `Запрос эмбеддингов Ollama упал: ${error instanceof Error ? error.message : String(error)}`,
    );
    return hashEmbed(preparedText, model, isQuery);
  }
}

export async function isModelAvailable(model: string): Promise<boolean> {
  if (model === 'localmind-hash-embed-v1') return true;
  return checkOllamaModelAvailable(model);
}

export function resetAvailabilityCache(): void {
  availabilityCache = null;
  hashFallbackWarned = false;
}
