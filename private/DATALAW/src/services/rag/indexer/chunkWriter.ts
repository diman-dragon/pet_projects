// src/services/rag/indexer/chunkWriter.ts
//
// FIX: убраны BEGIN/COMMIT вокруг writeChunksWithEmbeddings.
//
// Причина бага «cannot commit - no transaction is active»:
//   Tauri SQL plugin использует sqlx::SqlitePool. Между BEGIN и COMMIT стоит
//   await EmbeddingService.embed() — сетевой вызов к Ollama (100–3000ms).
//   Пока JS ждёт Ollama, соединение возвращается в пул и захватывается другим
//   запросом (UI, search, status update). Транзакция теряет владельца.
//   Когда управление возвращается, COMMIT вызывается на другом соединении → ошибка.
//
// Решение: НЕ оборачивать async-сетевые вызовы в SQL-транзакцию.
//   Каждый INSERT уже защищён своим withRetry. Атомарность на уровне документа
//   обеспечивается finishJob/failJob в indexSingleFile.
//
// [KG] Knowledge Graph: после записи каждого чанка вызываем KnowledgeGraphService.
//   Ошибки извлечения KG логируются, но НЕ прерывают индексацию.

import { EmbeddingService } from '../EmbeddingService';
import type { TextChunk } from '../ChunkService';
import { ensureVecEmbeddingsTable } from '../search/vecTable';
import { useOllamaStore } from '../../../store/ollamaStore';
import { withRetry } from '../../db/withRetry';
import { createLogger } from '../../logger/Logger';

const log = createLogger('chunkWriter');

export interface ChunkWriterOpts {
  db: any;
  libraryId: number;
  model: string;
  chunks: TextChunk[];
  jobId: number;
  signal?: AbortSignal;
  onProgress?: (done: number, total: number) => void;
  hasVec0: boolean;
  /** [KG] Опциональный контекст для Knowledge Graph */
  kgContext?: {
    clientId?: number;
    caseId?: number;
    /** Отключить KG-извлечение для этого документа (например, шаблоны) */
    skip?: boolean;
  };
}

/*
function extractArticleRef(text: string): string | null {
  const patterns = [
    /(?:ст(?:атья)?|art(?:icle)?|§)\s*\.?\s*(\d+(?:\.\d+)?(?:\s*[а-яёa-z])?(?:\s*п\.\s*\d+)?)/i,
    /Rule\s+(\d+)/i,
    /(?:Правило|п(?:ункт)?)\s*(\d+(?:\.\d+)?)/i,
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      return match[0].trim().slice(0, 80);
    }
  }

  return null;
}
*/

function getBatchSize(): number {
  const profile = useOllamaStore.getState().machineProfile;
  switch (profile) {
    case 'minimal':
      return 1;
    case 'light':
      return 2;
    case 'balanced':
    case 'full':
      return 3;
    default:
      return 2;
  }
}

/**
 * Пишет один чанк: сначала получаем embedding (async сеть),
 * затем синхронно вставляем в БД (без активной транзакции — чтобы не держать
 * соединение занятым во время сетевого вызова).
 *
 * [KG] Возвращает chunkId для последующего KG-извлечения.
 */
async function writeOneChunk(
  db: any,
  chunk: TextChunk,
  libraryId: number,
  model: string,
  hasVec0: boolean,
  parentDbId: number | null = null,
  skipEmbedding = false,
): Promise<number> {
  // ── Шаг 1: INSERT chunk (без embedding) ──────────────────────────────────
  // article_ref теперь приходит из Rust в самом чанке
  const chunkRes = await withRetry<{ lastInsertId: number | bigint }>(
    () =>
      db.execute(
        `INSERT INTO chunks (library_id, chunk_index, text, char_start, char_end, article_ref, parent_id)
       VALUES ($1,$2,$3,$4,$5,$6,$7)`,
        [
          libraryId,
          chunk.chunkIndex,
          chunk.text,
          chunk.charStart,
          chunk.charEnd,
          chunk.articleRef ?? null,
          parentDbId,
        ],
      ),
    { maxAttempts: 3, baseDelayMs: 50 },
  );

  const rawId = chunkRes.lastInsertId;
  const chunkId = typeof rawId === 'bigint' ? Number(rawId) : Math.trunc(Number(rawId));
  if (!Number.isInteger(chunkId) || chunkId < 1 || chunkId > 2_147_483_647) {
    throw new Error(`Invalid chunkId: ${rawId} (got ${chunkId})`);
  }

  if (skipEmbedding) {
    return chunkId;
  }

  // ── Шаг 2: Получаем embedding (сетевой вызов → не в транзакции!) ─────────
  const embedding = await EmbeddingService.embed(chunk.text, model, false);
  const vectorJson = JSON.stringify(Array.from(embedding.vector));

  // ── Шаг 3: INSERT embedding (JSON, для JS cosine search) ─────────────────
  await withRetry(
    () =>
      db.execute(
        `INSERT INTO embeddings (chunk_id, library_id, vector_blob, model, dim)
       VALUES ($1,$2,$3,$4,$5)`,
        [chunkId, libraryId, vectorJson, model, embedding.dim],
      ),
    { maxAttempts: 3, baseDelayMs: 50 },
  );

  // ── Шаг 4: INSERT в vec_embeddings (sqlite-vec virtual table) ────────────
  if (hasVec0) {
    await ensureVecEmbeddingsTable(db, embedding.dim);
    await withRetry(
      () =>
        db.execute(
          `INSERT INTO vec_embeddings (chunk_id, library_id, model, embedding)
         VALUES (CAST($1 AS INTEGER), CAST($2 AS INTEGER), $3, $4)`,
          [chunkId, libraryId, model, vectorJson],
        ),
      { maxAttempts: 3, baseDelayMs: 50 },
    );
  }

  return chunkId;
}

/**
 * Записывает все чанки документа с эмбеддингами.
 *
 * Намеренно НЕ использует BEGIN/COMMIT на весь батч: см. комментарий вверху файла.
 * Атомарность гарантируется на уровне документа через finishJob/failJob.
 *
 * [KG] После записи каждого чанка асинхронно запускает KG-извлечение.
 *       Ошибки KG не прерывают основную индексацию.
 */
export async function writeChunksWithEmbeddings(opts: ChunkWriterOpts): Promise<number> {
  const { db, libraryId, model, chunks, signal, onProgress, hasVec0, kgContext } = opts;
  let done = 0;
  const batchSize = getBatchSize();
  const skipKg = kgContext?.skip === true;
  let kgQueue: Promise<void> = Promise.resolve();

  log.info(
    `[📝 CHUNK-WRITE] Writing ${chunks.length} chunks (batchSize=${batchSize}, kg=${!skipKg})`,
  );

  // Карта для parent-child связи: chunkIndex (в текущем массиве) -> chunkId (в БД)
  const indexToDbId = new Map<number, number>();

  // Если в документе есть хотя бы один чанк с parentIndex, значит используется Parent-Child стратегия.
  // В этом случае эмбеддинги считаются ТОЛЬКО для дочерних чанков (у которых есть parentIndex).
  const isParentChildDoc = chunks.some((c) => c.parentIndex !== undefined);

  for (let index = 0; index < chunks.length; index += batchSize) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

    const batch = chunks.slice(index, index + batchSize);

    // Каждый chunk пишется последовательно — без общей транзакции
    for (const chunk of batch) {
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      try {
        const parentDbId =
          chunk.parentIndex !== undefined ? indexToDbId.get(chunk.parentIndex) : null;

        // В Parent-Child документах пропускаем эмбеддинги для родительских чанков (parentIndex === undefined)
        const skipEmbedding = isParentChildDoc && chunk.parentIndex === undefined;

        const chunkId = await writeOneChunk(
          db,
          chunk,
          libraryId,
          model,
          hasVec0,
          parentDbId ?? undefined,
          skipEmbedding,
        );
        indexToDbId.set(chunk.chunkIndex, chunkId);

        // [KG] Динамический импорт — избегаем circular deps и держим извлечение
        // в последовательной очереди, чтобы индекс считался готовым только после KG.
        if (!skipKg && !skipEmbedding) {
          // KG только для чанков с эмбеддингами
          kgQueue = kgQueue
            .then(() => import('../../knowledgeGraph/KnowledgeGraphService'))
            .then(({ KnowledgeGraphService }) =>
              KnowledgeGraphService.extractFromChunk(chunk.text, chunkId, libraryId, {
                clientId: kgContext?.clientId,
                caseId: kgContext?.caseId,
              }),
            )
            .then(() => undefined)
            .catch((err) => {
              log.warn(`[KG] extractFromChunk failed for chunk ${chunkId}`, err);
            });
        }
      } catch (err) {
        log.error(`[❌ CHUNK-WRITE] Failed at chunk ${chunk.chunkIndex}`, err);
        throw err; // пробрасываем — indexSingleFile вызовет failJob
      }
    }

    done += batch.length;
    onProgress?.(done, chunks.length);

    // Небольшая пауза между батчами, не перегружаем Ollama
    if (index + batchSize < chunks.length) {
      await sleep(20);
    }
  }

  if (!skipKg) {
    await kgQueue;
  }
  log.info(`[✅ CHUNK-WRITE] Done: ${done}/${chunks.length} chunks`);

  return done;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
