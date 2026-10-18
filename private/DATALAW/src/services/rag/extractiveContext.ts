// src/services/rag/extractiveContext.ts
//
// Extractive Context Selection — NO LLM
//
// ИЗМЕНЕНИЯ v2:
// 1. SOURCE DIVERSITY: максимум 2 чанка из одного файла в финальном контексте.
//    Это главный фикс "GOVERNANCE flooding" — один файл больше не может занять
//    все 5 слотов контекста.
// 2. RELEVANCE FLOOR: чанки с fastRerankScore < 0.05 пропускаются если есть
//    хотя бы один чанк с более высоким score (не выбрасываем все, только падаем аккуратно).
// 3. MAX_TOKENS увеличен до 3200 (был 1200) — вмещает 2-3 полных чанка.

import { createLogger } from '../logger/Logger';
import type { RetrievalCandidate } from './hybridRetrievalService';

const log = createLogger('ExtractiveContext');

const MAX_CHUNKS = 10;
const MAX_TOKENS = 6000;
const MAX_CHUNKS_PER_FILE = 4; // SOURCE DIVERSITY: не более 4 чанков из одного файла
const RELEVANCE_FLOOR = 0.04;  // Чанки ниже этого порога — только если лучших нет

export interface ExtractedContext {
  chunks: RetrievalCandidate[];
  totalTokens: number;
  chunkCount: number;
}

export function extractContext(
  candidates: RetrievalCandidate[],
  _query: string,
): ExtractedContext {
  if (candidates.length === 0) {
    return { chunks: [], totalTokens: 0, chunkCount: 0 };
  }

  // Считаем максимальный score для relative floor
  const maxScore = Math.max(...candidates.map(c =>
    c.llmRerankScore ?? c.fastRerankScore ?? c.rrfScore ?? 0
  ));

  // Фильтруем очень слабые чанки если есть достаточно хороших
  const hasGoodChunks = candidates.some(c =>
    (c.llmRerankScore ?? c.fastRerankScore ?? 0) >= RELEVANCE_FLOOR
  );

  const eligibleCandidates = hasGoodChunks
    ? candidates.filter(c => (c.llmRerankScore ?? c.fastRerankScore ?? c.rrfScore ?? 0) >= RELEVANCE_FLOOR)
    : candidates;

  // Если используется Parent-Child стратегия, подменяем текст чанка на текст родителя
  const mappedCandidates = eligibleCandidates.map(c => ({
    ...c,
    chunkText: c.parentText || c.chunkText
  }));

  // SOURCE DIVERSITY: группируем по файлу, берём не более MAX_CHUNKS_PER_FILE из каждого
  const fileChunkCount = new Map<string, number>();
  const diverseChunks: RetrievalCandidate[] = [];

  for (const chunk of mappedCandidates) {
    if (diverseChunks.length >= MAX_CHUNKS) break;

    const filePath = chunk.filePath || 'unknown';
    const countForFile = fileChunkCount.get(filePath) ?? 0;

    if (countForFile >= MAX_CHUNKS_PER_FILE) continue; // Пропускаем: этот файл уже достаточно представлен

    fileChunkCount.set(filePath, countForFile + 1);
    diverseChunks.push(chunk);
  }

  // Если после diversity-фильтра чанков мало — добираем из любых источников
  if (diverseChunks.length < 2 && mappedCandidates.length > diverseChunks.length) {
    const alreadyAdded = new Set(diverseChunks.map(c => c.chunkId));
    for (const chunk of mappedCandidates) {
      if (diverseChunks.length >= MAX_CHUNKS) break;
      if (alreadyAdded.has(chunk.chunkId)) continue;
      diverseChunks.push(chunk);
      alreadyAdded.add(chunk.chunkId);
    }
  }

  // Применяем лимит токенов
  const extracted: RetrievalCandidate[] = [];
  let totalTokens = 0;

  for (const chunk of diverseChunks) {
    // [KG] Citations added
    const citedText = `[ИСТОЧНИК: ${chunk.title}${chunk.articleRef ? `, ст. ${chunk.articleRef}` : ''}] ${chunk.chunkText}`;
    const chunkTokens = estimateTokens(citedText);

    if (totalTokens + chunkTokens > MAX_TOKENS) {
      const remainingTokens = MAX_TOKENS - totalTokens;
      if (remainingTokens >= 200) {
        const shortened = citedText.slice(0, remainingTokens * 4);
        extracted.push({ ...chunk, chunkText: shortened });
        totalTokens += estimateTokens(shortened);
      }
      break;
    }

    extracted.push({ ...chunk, chunkText: citedText });
    totalTokens += chunkTokens;
  }

  log.debug('Context extracted', {
    inputChunks: candidates.length,
    eligibleChunks: eligibleCandidates.length,
    diverseChunks: diverseChunks.length,
    outputChunks: extracted.length,
    uniqueFiles: fileChunkCount.size,
    totalTokens,
    maxScore: maxScore.toFixed(4),
  });

  return {
    chunks: extracted,
    totalTokens,
    chunkCount: extracted.length,
  };
}

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

