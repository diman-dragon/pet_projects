// src/services/search/DocumentIndexer.ts
//
// Единая точка индексации документов.
//
// БЫСТРЫЙ РЕЖИМ (по умолчанию):
//   1. Извлечение текста
//   2. Детекция языка
//   3. Категоризация по пути
//   4. Upsert в library + lang + folder_path
//   5. library_fts обновляется автоматически через триггеры (итерация 1)
//
// ФОНОВЫЙ РЕЖИМ (если skipEmbeddings = false):
//   После быстрого режима — chunks, embeddings, vec_embeddings, index_jobs.
//   Не блокирует поиск. Управляется сигналом отмены.
//
// Использование:
//   const result = await DocumentIndexer.index({ filePath, signal });
//   const result = await DocumentIndexer.index({ filePath, skipEmbeddings: true }); // только FTS

import { ConnectionManager } from '../db/ConnectionManager';
import { EmbeddingModelService } from '../ai/EmbeddingModelService';
import { TextExtractorService } from '../rag/TextExtractorService';
import { ChunkService } from '../rag/ChunkService';
import { detectCapability } from '../rag/search/capability';
import { writeChunksWithEmbeddings } from '../rag/indexer/chunkWriter';
import { hashText } from '../rag/indexer/hashText';
import { insertPlainChunks } from '../rag/indexer/insertPlainChunks';
import { createJob, failJob, finishJob } from '../rag/indexer/jobTracker';
import { upsertLibrary } from '../rag/indexer/libraryUpsert';
import { titleFromPath } from '../rag/indexer/titleFromPath';
import { isTsvText, splitTsvRecords } from '../rag/textExtractor/normalizeTsvText';
import { detectLang } from './languageDetection';
import { FolderCategorizerService } from './FolderCategorizerService';
import { createLogger } from '../logger/Logger';
import type { TextChunk } from '../rag/ChunkService';

const log = createLogger('DocumentIndexer');

export interface DocumentIndexOptions {
  filePath: string;
  title?: string;
  category?: string;
  metadata?: Record<string, unknown>;
  /** Пропустить chunks/embeddings — только library + FTS (быстрый режим) */
  skipEmbeddings?: boolean;
  onProgress?: (done: number, total: number) => void;
  signal?: AbortSignal;
}

export interface DocumentIndexResult {
  libraryId: number;
  chunkCount: number;
  skipped: boolean;
  lang: string | null;
  category: string;
  warning?: string;
}

function buildTsvChunks(normalizedText: string): TextChunk[] {
  const records = splitTsvRecords(normalizedText);
  let charPos = 0;
  return records.map((record, idx) => {
    const chunk: TextChunk = {
      text: record,
      chunkIndex: idx,
      charStart: charPos,
      charEnd: charPos + record.length,
    };
    charPos += record.length + 2;
    return chunk;
  });
}

async function indexDocument(opts: DocumentIndexOptions): Promise<DocumentIndexResult> {
  const {
    filePath,
    title,
    category,
    metadata = {},
    skipEmbeddings = false,
    onProgress,
    signal,
  } = opts;

  const db = ConnectionManager.get('kb_main');
  const model = await EmbeddingModelService.get();

  // ── 1. Извлечение текста ────────────────────────────────────────────────────
  const extracted = await TextExtractorService.extract(filePath);
  if (!extracted.text || extracted.text.length < 10) {
    return {
      libraryId: -1,
      chunkCount: 0,
      skipped: true,
      lang: null,
      category: category ?? 'General',
      warning: extracted.warning ?? 'Не удалось извлечь текст',
    };
  }

  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

  // ── 2. Детекция языка ───────────────────────────────────────────────────────
  const lang = detectLang(extracted.text, null);

  // ── 3. Категоризация по пути ────────────────────────────────────────────────
  const folderPath = FolderCategorizerService.extractFolderPath(filePath);
  const resolvedCategory = category
    ?? await FolderCategorizerService.categorizeByPath(filePath, null, null);

  // ── 4. Upsert в library (FTS обновляется триггером) ─────────────────────────
  const hash = await hashText(extracted.text);
  const docTitle = title ?? titleFromPath(filePath);
  const fileType = filePath.split('.').pop()?.toLowerCase() ?? 'txt';

  const upsert = await upsertLibrary(
    db,
    filePath,
    docTitle,
    resolvedCategory,
    fileType,
    extracted.text,
    hash,
    metadata,
    model,
    lang,
    folderPath,
  );

  if (!upsert.needsReindex) {
    return { libraryId: upsert.libraryId, chunkCount: 0, skipped: true, lang, category: resolvedCategory };
  }

  // ── 5. Быстрый режим: только FTS ───────────────────────────────────────────
  if (skipEmbeddings) {
    const isTsv = isTsvText(extracted.text);
    const chunks = isTsv
      ? buildTsvChunks(extracted.text)
      : await ChunkService.chunk(extracted.text);

    if (chunks.length) {
      await insertPlainChunks(db, upsert.libraryId, chunks);
    }

    log.info(`Fast-indexed "${docTitle}" [lang=${lang ?? '?'}] cat=${resolvedCategory}`);
    return { libraryId: upsert.libraryId, chunkCount: chunks.length, skipped: false, lang, category: resolvedCategory };
  }

  // ── 6. Фоновый режим: chunks + embeddings ──────────────────────────────────
  const isTsv = isTsvText(extracted.text);
  const chunks = isTsv
    ? buildTsvChunks(extracted.text)
    : await ChunkService.chunk(extracted.text);

  if (!chunks.length) {
    return {
      libraryId: upsert.libraryId,
      chunkCount: 0,
      skipped: false,
      lang,
      category: resolvedCategory,
      warning: 'Текст слишком короткий',
    };
  }

  const capability = await detectCapability(db);
  const jobId = await createJob(db, upsert.libraryId, chunks.length);

  try {
    const done = await writeChunksWithEmbeddings({
      db,
      libraryId: upsert.libraryId,
      model,
      chunks,
      jobId,
      signal,
      onProgress,
      hasVec0: capability === 'vec0',
    });

    await finishJob(db, jobId);
    log.info(`Full-indexed "${docTitle}" [lang=${lang ?? '?'}] ${done} chunks (${isTsv ? 'TSV' : 'text'})`);
    return { libraryId: upsert.libraryId, chunkCount: done, skipped: false, lang, category: resolvedCategory };
  } catch (error: any) {
    const message = error?.message ?? String(error);
    await failJob(db, jobId, message).catch(() => {});
    throw error;
  }
}

export const DocumentIndexer = {
  index: indexDocument,

  /** Быстрый режим: только library + FTS, без embeddings */
  async indexFast(opts: Omit<DocumentIndexOptions, 'skipEmbeddings'>): Promise<DocumentIndexResult> {
    return indexDocument({ ...opts, skipEmbeddings: true });
  },

  /** Полный режим: library + FTS + chunks + embeddings */
  async indexFull(opts: Omit<DocumentIndexOptions, 'skipEmbeddings'>): Promise<DocumentIndexResult> {
    return indexDocument({ ...opts, skipEmbeddings: false });
  },
};
