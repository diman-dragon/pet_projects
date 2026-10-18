// src/services/rag/search/vecScalar.ts
//
// FIX: нормализация имён моделей в SQL WHERE e.model IN (...).
// 'bge-m3:latest' → ищем также 'bge-m3' и наоборот.

import { createLogger } from '../../logger/Logger';
import type { SearchResult } from '../UnifiedSearchService';

const log = createLogger('vecScalarSearch');

function normalizeModelName(model: string): string {
  return model.replace(/:latest$/i, '').trim();
}

export async function searchVecScalar(
  db: any,
  queryVec: Float32Array,
  limit: number,
  category: string | undefined,
  model: string | string[],
): Promise<SearchResult[]> {
  try {
    const vecJson = JSON.stringify(Array.from(queryVec));
    const catParam = category && category !== 'all' ? category : null;
    const modelList = Array.isArray(model) ? model : [model];

    // FIX: оба варианта имени — с тегом и без
    const expandedModels = [...new Set(
      modelList.flatMap((m) => {
        const base = normalizeModelName(m);
        return [m, base, base + ':latest'];
      }),
    )];

    const placeholders = expandedModels.map((_, idx) => `\$${idx + 2}`).join(', ');
    const catIdx = expandedModels.length + 2;
    const limitIdx = expandedModels.length + 3;
    const params: (string | null | number)[] = [vecJson, ...expandedModels, catParam, limit];

    const rows = (await db.select(
      `SELECT c.id AS chunk_id, c.library_id, c.chunk_index, c.text AS chunk_text,
              c.char_start, c.char_end, c.article_ref, c.parent_id,
              p.text AS parent_text,
              l.title, l.category, l.file_path, l.lang,
              (1.0 - vec_distance_cosine(e.vector_blob, $1)) AS score
       FROM embeddings e
       JOIN chunks c ON c.id = e.chunk_id
       JOIN library l ON l.id = e.library_id
       LEFT JOIN chunks p ON p.id = c.parent_id
       WHERE e.model IN (${placeholders})
         AND ($${catIdx} IS NULL OR l.category = $${catIdx})
       ORDER BY score DESC
       LIMIT $${limitIdx}`,
      params,
    )) as any[];

    if (!rows?.length) return [];

    return rows.map((r: any) => ({
      chunkId: r.chunk_id as number,
      libraryId: r.library_id as number,
      title: r.title as string,
      category: r.category as string,
      filePath: r.file_path as string,
      chunkText: r.chunk_text as string,
      chunkIndex: r.chunk_index as number,
      score: Math.min(Math.max(Number(r.score), 0), 1),
      matchType: 'vector' as const,
      lang: r.lang as string | undefined,
      charStart: r.char_start as number | undefined,
      charEnd: r.char_end as number | undefined,
      articleRef: r.article_ref as string | null | undefined,
      parentId: r.parent_id as number | undefined,
      parentText: r.parent_text as string | undefined,
    }));
  } catch (error) {
    log.error('vecScalar search failed:', error);
    return [];
  }
}
