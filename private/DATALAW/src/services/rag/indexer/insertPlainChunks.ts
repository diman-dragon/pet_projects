import type { DatabaseLike } from '../../db/ConnectionManager';
import type { TextChunk } from '../ChunkService';
import { withRetry } from '../../db/withRetry';
import { createLogger } from '../../logger/Logger';

const log = createLogger('insertPlainChunks');

const CHUNK_INSERT_BATCH = 50;

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

export async function insertPlainChunks(
  db: DatabaseLike,
  libraryId: number,
  chunks: TextChunk[],
): Promise<void> {
  if (chunks.length === 0) return;

  const batches: TextChunk[][] = [];
  for (let i = 0; i < chunks.length; i += CHUNK_INSERT_BATCH) {
    batches.push(chunks.slice(i, i + CHUNK_INSERT_BATCH));
  }

  log.info(`[CHUNKS] Inserting ${chunks.length} chunks in ${batches.length} batches`);

  for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
    const batch = batches[batchIndex];

    await withRetry(
      async () => {
        await db.execute('BEGIN TRANSACTION');

        try {
          for (const chunk of batch) {
            await db.execute(
              `INSERT INTO chunks (library_id, chunk_index, text, char_start, char_end, article_ref)
               VALUES ($1,$2,$3,$4,$5,$6)`,
              [libraryId, chunk.chunkIndex, chunk.text, chunk.charStart, chunk.charEnd, extractArticleRef(chunk.text)],
            );
          }

          await db.execute('COMMIT');
        } catch (error) {
          try {
            await db.execute('ROLLBACK');
          } catch {
            // Ignore rollback failures so the original DB error is preserved.
          }
          throw error;
        }
      },
      {
        maxAttempts: 5,
        baseDelayMs: 100,
        maxDelayMs: 3000,
        onRetry: (attempt, delay) => {
          log.warn(`[CHUNKS] Batch ${batchIndex + 1}/${batches.length} busy, retry ${attempt} in ${delay}ms`);
        },
      },
    );

    if (batchIndex < batches.length - 1) {
      await sleep(10);
    }
  }

  log.info(`[CHUNKS] Successfully inserted ${chunks.length} chunks`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
