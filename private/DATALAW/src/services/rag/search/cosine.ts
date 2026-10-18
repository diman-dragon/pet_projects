// src/services/rag/search/cosine.ts
//
// FIX: нормализация имён моделей в SQL WHERE e.model IN (...).
//
// Проблема: если документы индексировались с model='bge-m3' а запрос приходит
// с model='bge-m3:latest' — WHERE e.model IN ('bge-m3:latest') не находит строк.
// cosine JS-поиск возвращал [] → полный провал всех трёх методов поиска.
//
// Фикс: для каждого запрошенного имени добавляем обе формы в SQL (с :latest и без).
// MIN_COSINE_SCORE снижен до 0.08 — соответствует новому vectorMin в config.ts (0.10).
// VectorSearchService применяет свой порог поверх, поэтому здесь безопасно быть мягче.

import { EmbeddingService } from '../EmbeddingService';
import { createLogger } from '../../logger/Logger';
import type { SearchResult } from '../UnifiedSearchService';

const log = createLogger('cosineSearch');

// FIX: снижен с 0.10 → 0.08 — VectorSearchService применяет vectorMin (0.10) поверх
const MIN_COSINE_SCORE = 0.08;

function normalizeModelName(model: string): string {
  return model.replace(/:latest$/i, '').trim();
}

function dotProduct(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

export async function searchJsCosine(
  db: any,
  queryVec: Float32Array,
  limit: number,
  category: string | undefined,
  model: string | string[],
): Promise<SearchResult[]> {
  try {
    const catParam = category && category !== 'all' ? category : null;
    const modelList = Array.isArray(model) ? model : [model];

    // FIX: для каждого имени добавляем обе формы — с тегом и без
    // 'bge-m3:latest' → ['bge-m3:latest', 'bge-m3']
    const expandedModels = [...new Set(
      modelList.flatMap((m) => {
        const base = normalizeModelName(m);
        return [m, base, base + ':latest'];
      }),
    )];

    const placeholders = expandedModels.map((_, idx) => `$${idx + 1}`).join(', ');
    const params: (string | null)[] = [...expandedModels, catParam];

    const rows = (await db.select(
      `SELECT e.chunk_id, e.library_id, c.chunk_index, c.text AS chunk_text,
              c.char_start, c.char_end, c.article_ref,
              e.vector_blob, l.title, l.category, l.file_path, l.lang
       FROM embeddings e
       JOIN chunks  c ON c.id = e.chunk_id
       JOIN library l ON l.id = e.library_id
       WHERE e.model IN (${placeholders})
         AND ($${expandedModels.length + 1} IS NULL OR l.category = $${expandedModels.length + 1})`,
      params,
    )) as any[];

    if (!rows?.length) {
      log.warn(`cosine JS: 0 rows for models ${JSON.stringify(expandedModels)}`);
      return [];
    }

    const results: SearchResult[] = [];
    for (const row of rows) {
      const vec = EmbeddingService.deserialize(row.vector_blob);
      if (!vec || vec.length === 0) continue;

      const score = dotProduct(queryVec, vec);
      if (score < MIN_COSINE_SCORE) continue;

      results.push({
        chunkId: row.chunk_id as number,
        libraryId: row.library_id as number,
        title: row.title as string,
        category: row.category as string,
        filePath: row.file_path as string,
        chunkText: row.chunk_text as string,
        chunkIndex: row.chunk_index as number,
        score,
        matchType: 'vector' as const,
        lang: row.lang as string | undefined,
        charStart: row.char_start as number | undefined,
        charEnd: row.char_end as number | undefined,
        articleRef: row.article_ref as string | null | undefined,
      });
    }

    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  } catch (error) {
    log.error('JS cosine search failed:', error);
    return [];
  }
}
