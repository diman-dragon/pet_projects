// src/services/rag/indexer/indexSingleFile.ts
//
// ИЗМЕНЕНИЯ v4 (fix7):
// 1. Rust legalChunkText (rag_chunk_legal) вызывается ПЕРВЫМ для всех нон-TSV файлов.
//    Если Rust вернул [] (Generic) — падаем на ChunkService.chunk() (Rust rag_chunk_text).
//    TS legalChunker.ts УДАЛЁН — вся логика структурного чанкинга в Rust.
// 2. Стратегия выбора:
//    TSV/CSV  → buildTsvChunks (1 запись = 1 чанк)
//    Всё остальное → rag_chunk_legal (Rust)
//      └── если [] → rag_chunk_text (Rust generic fallback)
// 3. TOC-фильтр применяется после любой стратегии.

import { EmbeddingModelService } from '../../ai/EmbeddingModelService';
import { ConnectionManager } from '../../db/ConnectionManager';
import { createLogger } from '../../logger/Logger';
import { ChunkService } from '../ChunkService';
import { TextExtractorService } from '../TextExtractorService';
import { detectCapability } from '../search/capability';
import { writeChunksWithEmbeddings } from './chunkWriter';
import { hashText } from './hashText';
import { insertPlainChunks } from './insertPlainChunks';
import { createJob, failJob, finishJob } from './jobTracker';
import { upsertLibrary } from './libraryUpsert';
import { titleFromPath } from './titleFromPath';
import { isTsvText, splitTsvRecords } from '../textExtractor/normalizeTsvText';
import { detectLang } from '../../search/languageDetection';
import { FolderCategorizerService } from '../../search/FolderCategorizerService';
import { legalChunkText } from '../native/legalChunkText';
import { getRagConfig } from '../config';
import type { IndexOptions, IndexResult } from './types';
import type { TextChunk } from '../ChunkService';

const log = createLogger('IndexerService');

function buildTsvChunks(normalizedText: string): TextChunk[] {
  const records = splitTsvRecords(normalizedText);
  let charPos = 0;
  return records.map((record, idx) => {
    const chunk: TextChunk = { text: record, chunkIndex: idx, charStart: charPos, charEnd: charPos + record.length };
    charPos += record.length + 2;
    return chunk;
  });
}

function isTocChunk(text: string): boolean {
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
  if (lines.length === 0) return true;
  const dotLines = lines.filter(l => /\.{5,}/.test(l));
  if (dotLines.length / lines.length > 0.4) return true;
  const meaningful = text.replace(/[\d\s.…·•\-–—\/\\|]+/g, '');
  return meaningful.length < 20 && text.length > 10;
}

function filterChunks(chunks: TextChunk[]): TextChunk[] {
  return chunks
    .filter(c => !isTocChunk(c.text))
    .map((c, i) => ({ ...c, chunkIndex: i }));
}

export async function indexSingleFile(opts: IndexOptions): Promise<IndexResult> {
  const { filePath, title, category, metadata = {}, onProgress, signal, skipEmbeddings } = opts;

  const db = ConnectionManager.get('kb_main');
  const model = await EmbeddingModelService.get();
  const extracted = await TextExtractorService.extract(filePath);

  if (!extracted.text || extracted.text.length < 10) {
    return { libraryId: -1, chunkCount: 0, skipped: true, warning: extracted.warning ?? 'Не удалось извлечь текст' };
  }

  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

  const hash = await hashText(extracted.text);
  const docTitle = title ?? titleFromPath(filePath);
  const fileType = filePath.split('.').pop()?.toLowerCase() ?? 'txt';
  const resolvedCategory = category ?? await FolderCategorizerService.categorizeByPath(filePath, null, null);
  const folderPath = FolderCategorizerService.extractFolderPath(filePath);
  const lang = detectLang(extracted.text, null);
  const config = getRagConfig();

  const upsert = await upsertLibrary(db, filePath, docTitle, resolvedCategory, fileType,
    extracted.text, hash, metadata, model, lang, folderPath);

  if (!upsert.needsReindex) return { libraryId: upsert.libraryId, chunkCount: 0, skipped: true };

  // ── Выбор стратегии чанкинга ────────────────────────────────────────────
  let chunks: TextChunk[];
  let strategy: string;

  if (isTsvText(extracted.text)) {
    strategy = 'tsv';
    chunks = buildTsvChunks(extracted.text);
    log.info(`[Indexer] strategy=TSV → ${chunks.length} chunks`);
  } else {
    // Пробуем Rust юридический чанкер
    const legalResult = await legalChunkText(extracted.text, {
      docTitle,
      maxChunkChars: config.chunking.chunkSize,
      minChunkChars: config.chunking.minChunkSize,
      childSize: config.chunking.childSize,
    });

    if (legalResult.chunks.length > 0) {
      strategy = `legal:${legalResult.docType}`;
      chunks = filterChunks(legalResult.chunks);
      log.info(`[Indexer] strategy=${strategy} → ${legalResult.chunks.length} raw → ${chunks.length} chunks`);
    } else {
      // Generic fallback — Rust посимвольный чанкер
      strategy = 'generic';
      const rawChunks = await ChunkService.chunk(extracted.text);
      chunks = filterChunks(rawChunks);
      log.info(`[Indexer] strategy=generic → ${chunks.length} chunks`);
    }
  }

  if (!chunks.length) {
    return { libraryId: upsert.libraryId, chunkCount: 0, skipped: false, warning: 'Нет чанков' };
  }

  if (skipEmbeddings) {
    await insertPlainChunks(db, upsert.libraryId, chunks);
    return { libraryId: upsert.libraryId, chunkCount: chunks.length, skipped: false };
  }

  const capability = await detectCapability(db);
  const jobId = await createJob(db, upsert.libraryId, chunks.length);

  try {
    const done = await writeChunksWithEmbeddings({
      db, libraryId: upsert.libraryId, model, chunks, jobId, signal, onProgress,
      hasVec0: capability === 'vec0',
    });
    await finishJob(db, jobId);
    log.info(`[Indexer] Done: "${docTitle}" [lang=${lang ?? '?'}, ${strategy}] ${done} chunks`);
    return { libraryId: upsert.libraryId, chunkCount: done, skipped: false };
  } catch (error: any) {
    await failJob(db, jobId, error?.message ?? String(error)).catch(() => {});
    throw error;
  }
}
