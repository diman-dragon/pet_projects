// src/services/rag/search/vec0.ts
//
// FIX: нормализация имени модели при JS-фильтрации.
//
// Проблема: документы могли быть проиндексированы с именем 'bge-m3' (без :latest),
// а EmbeddingModelService возвращает 'bge-m3:latest'.
// vec0 KNN возвращает строки с row.model='bge-m3', но acceptedModels=['bge-m3:latest'] →
// filter = 0 строк → тихий fallback на FTS → пользователь видит только keyword-результаты.
//
// Решение: нормализовать имена моделей перед сравнением (убрать :latest).
// Если после нормализации всё равно нет совпадений — берём топ без фильтра модели
// (в базе заведомо только один тип эмбеддингов).

import { createLogger } from '../../logger/Logger';
import type { SearchResult } from '../UnifiedSearchService';

const log = createLogger('vec0Search');

function normalizeModelName(model: string): string {
  return model.replace(/:latest$/i, '').trim().toLowerCase();
}

export async function searchVec0(
  db: any,
  queryVec: Float32Array,
  limit: number,
  category: string | undefined,
  model: string | string[],
): Promise<SearchResult[]> {
  try {
    const vecJson = JSON.stringify(Array.from(queryVec));
    const knn = limit * 4;

    const knnRows = (await db.select(
      `SELECT chunk_id, library_id, model, distance
       FROM vec_embeddings
       WHERE embedding MATCH $1
         AND k = $2`,
      [vecJson, knn],
    )) as Array<{ chunk_id: number; library_id: number; model: string; distance: number }>;

    if (!knnRows?.length) return [];

    const acceptedModels = Array.isArray(model) ? model : [model];
    // FIX: нормализуем имена — сравниваем без суффиксов тегов (:latest)
    const acceptedNormalized = new Set(acceptedModels.map(normalizeModelName));

    let filtered = knnRows.filter((r) =>
      acceptedNormalized.has(normalizeModelName(r.model)),
    );

    if (!filtered.length) {
      const foundModels = [...new Set(knnRows.map((r) => r.model))].join(', ');
      log.warn(
        `vec0: model filter mismatch. Wanted: ${JSON.stringify(acceptedModels)}, ` +
        `found in vec_embeddings: ${foundModels}. Using unfiltered results.`,
      );
      // FIX: fallback на нефильтрованные результаты вместо []
      // В типичной инсталляции один тип эмбеддингов → безопасно
      filtered = knnRows;
    }

    return fetchChunkDetails(db, filtered.slice(0, limit), category);
  } catch (error) {
    log.error('vec0 search failed:', error);
    return [];
  }
}

async function fetchChunkDetails(
  db: any,
  rows: Array<{ chunk_id: number; distance: number }>,
  category: string | undefined,
): Promise<SearchResult[]> {
  if (!rows.length) return [];

  const chunkIds = rows.map((r) => r.chunk_id);
  const placeholders = chunkIds.map((_, i) => `$${i + 1}`).join(',');
  const catParam = category && category !== 'all' ? category : null;

  const detailRows = (await db.select(
    `SELECT c.id AS chunk_id, c.library_id, c.chunk_index, c.text AS chunk_text,
            c.char_start, c.char_end, c.article_ref, c.parent_id,
            p.text AS parent_text,
            l.title, l.category, l.file_path, l.lang
     FROM chunks c
     JOIN library l ON l.id = c.library_id
     LEFT JOIN chunks p ON p.id = c.parent_id
     WHERE c.id IN (${placeholders})
       AND ($${chunkIds.length + 1} IS NULL OR l.category = $${chunkIds.length + 1})`,
    [...chunkIds, catParam],
  )) as any[];

  if (!detailRows?.length) return [];

  const distanceMap = new Map(rows.map((r) => [r.chunk_id, r.distance]));

  return detailRows
    .map((r: any) => {
      const distance = distanceMap.get(r.chunk_id as number) ?? 2;
      const score = Math.min(Math.max(1.0 - (distance * distance) / 2.0, 0), 1);
      return {
        chunkId: r.chunk_id as number,
        libraryId: r.library_id as number,
        title: r.title as string,
        category: r.category as string,
        filePath: r.file_path as string,
        chunkText: r.chunk_text as string,
        chunkIndex: r.chunk_index as number,
        score,
        matchType: 'vector' as const,
        lang: r.lang as string | undefined,
        charStart: r.char_start as number | undefined,
        charEnd: r.char_end as number | undefined,
        articleRef: r.article_ref as string | null | undefined,
        parentId: r.parent_id as number | undefined,
        parentText: r.parent_text as string | undefined,
      };
    })
    .sort((a, b) => b.score - a.score);
}
