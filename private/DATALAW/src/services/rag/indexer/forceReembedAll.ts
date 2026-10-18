import { EmbeddingModelService } from '../../ai/EmbeddingModelService';
import { ConnectionManager } from '../../db/ConnectionManager';
import { createLogger } from '../../logger/Logger';
import { EmbeddingService } from '../EmbeddingService';
import { UnifiedSearchService } from '../UnifiedSearchService';
import { detectCapability } from '../search/capability';
import { ensureVecEmbeddingsTable } from '../search/vecTable';
import { CHUNKER_VERSION } from './libraryUpsert';

const log = createLogger('forceReembedAll');

export interface ReembedProgress {
  done: number;
  total: number;
  currentTitle: string;
  currentDocDone: number;
  currentDocTotal: number;
}

export interface ReembedResult {
  done: number;
  errors: string[];
  aborted: boolean;
  backendDown: boolean;
}

export async function forceReembedAll(
  onProgress?: (progress: ReembedProgress) => void,
  signal?: AbortSignal,
): Promise<ReembedResult> {
  const yieldEvery = 10;
  const db = ConnectionManager.get('kb_main');
  const model = await EmbeddingModelService.get();

  log.info(`forceReembedAll: model=${model}`);

  EmbeddingService.clearCache();
  UnifiedSearchService.clearCache();

  await db.execute(`DELETE FROM embeddings`);
  try {
    await db.execute(`DELETE FROM vec_embeddings`);
  } catch {
    // vec0 is optional.
  }

  const chunks = await db.select<{ id: number; library_id: number; text: string }[]>(
    `SELECT id, library_id, text FROM chunks ORDER BY library_id, id`,
  );
  const titleRows = await db.select<{ id: number; title: string }[]>(`SELECT id, title FROM library`);
  const titles = new Map(titleRows.map((item) => [item.id, item.title]));
  const capability = await detectCapability(db);
  const hasVec0 = capability === 'vec0';

  const chunksByDoc = new Map<number, typeof chunks>();
  for (const chunk of chunks) {
    if (!chunksByDoc.has(chunk.library_id)) chunksByDoc.set(chunk.library_id, []);
    chunksByDoc.get(chunk.library_id)!.push(chunk);
  }

  let done = 0;
  const errors: string[] = [];
  let backendDown = false;
  let consecutiveEmbedErrors = 0;
  const total = chunks.length;

  for (const docId of Array.from(chunksByDoc.keys())) {
    if (signal?.aborted || backendDown) break;

    const docChunks = chunksByDoc.get(docId)!;
    const docTitle = titles.get(docId) ?? `doc#${docId}`;
    let docDone = 0;

    for (const chunk of docChunks) {
      if (signal?.aborted || backendDown) break;

      onProgress?.({
        done,
        total,
        currentTitle: docTitle,
        currentDocDone: docDone,
        currentDocTotal: docChunks.length,
      });

      try {
        const embedding = await EmbeddingService.embed(chunk.text, model, false);
        const vectorJson = JSON.stringify(Array.from(embedding.vector));

        await db.execute(
          `INSERT OR REPLACE INTO embeddings (chunk_id, library_id, vector_blob, model, dim)
           VALUES ($1, $2, $3, $4, $5)`,
          [chunk.id, chunk.library_id, vectorJson, model, embedding.dim],
        );

        if (hasVec0) {
          await ensureVecEmbeddingsTable(db, embedding.dim);
          await db.execute(
            `INSERT OR REPLACE INTO vec_embeddings (chunk_id, library_id, model, embedding)
             VALUES (CAST($1 AS INTEGER), CAST($2 AS INTEGER), $3, $4)`,
            [chunk.id, chunk.library_id, model, vectorJson],
          );
        }

        consecutiveEmbedErrors = 0;
        done++;
        docDone++;

        if (done % yieldEvery === 0) {
          await new Promise((r) => setTimeout(r, 0));
        }
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        errors.push(`chunk ${chunk.id}: ${msg}`);
        log.error(`Embed error chunk ${chunk.id}:`, error);
        consecutiveEmbedErrors++;

        if (consecutiveEmbedErrors >= 5) {
          log.error('5 consecutive embed errors — Ollama likely down, aborting');
          backendDown = true;
          break;
        }
      }
    }
  }

  // Обновляем chunker_version для всех документов после успешной переиндексации
  if (done > 0 && !backendDown) {
    await db.execute(
      `UPDATE library SET chunker_version = $1 WHERE id IN (
        SELECT DISTINCT library_id FROM chunks
      )`,
      [CHUNKER_VERSION]
    );
    log.info(`Updated chunker_version to ${CHUNKER_VERSION} for all documents`);
  }

  UnifiedSearchService.clearCache();

  return {
    done,
    errors,
    aborted: signal?.aborted ?? false,
    backendDown,
  };
}
