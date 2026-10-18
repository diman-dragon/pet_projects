// src/services/search/QueryProcessor.ts
//
// РЕЭКСПОРТ из queryUnderstanding.ts
// QueryProcessor теперь является адаптером к understandQuery для обратной совместимости.
//
// ИЗМЕНЕНИЯ:
//   - processQuery теперь делегирует в understandQuery из rag/queryUnderstanding.ts
//   - Интерфейс ProcessedQuery сохранён для обратной совместимости
//   - Все legacy-функции (tokenization, stop-words, expansion) удалены

import {
  understandQuery,
  type QueryUnderstandingResult,
} from '../rag/queryUnderstanding';

// Legacy тип для обратной совместимости
export type QueryType = 'exact_identifier' | 'name' | 'date_range' | 'free_text';

// Legacy интерфейс для обратной совместимости
export interface ProcessedQuery {
  normalizedQuery: string;
  tokens: string[];
  queryType: QueryType;
  queryLang: string;
  expandedTerms: string[];
}

// ── Адаптер intent → queryType ───────────────────────────────────────────────

function intentToQueryType(intent: QueryUnderstandingResult['intent']): QueryType {
  switch (intent) {
    case 'legal':
      return 'free_text'; // legal запросы идут как free_text с расширением
    case 'procedural':
      return 'free_text';
    case 'biographical':
      return 'name'; // биографические → поиск по имени
    case 'conversational':
      return 'free_text'; // разговорные тоже free_text
    case 'factual':
    default:
      return 'free_text';
  }
}

// ── Упрощённая токенизация (без stop-words для совместимости) ────────────────

function simpleTokenize(query: string): string[] {
  return query
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 2)
    .slice(0, 10);
}

// ── Основная функция-адаптер ─────────────────────────────────────────────────

/**
 * Обрабатывает поисковый запрос.
 * Делегирует в understandQuery и адаптирует результат под ProcessedQuery.
 */
export async function processQuery(rawQuery: string): Promise<ProcessedQuery> {
  const understanding = await understandQuery(rawQuery);

  return {
    normalizedQuery: understanding.rewritten_query,
    tokens: simpleTokenize(understanding.rewritten_query),
    queryType: intentToQueryType(understanding.intent),
    queryLang: understanding.lang ?? 'ru',
    expandedTerms: understanding.expansions,
  };
}

// ── Warm-up (заглушка для совместимости) ─────────────────────────────────────

export async function warmupQueryExpansion(): Promise<void> {
  // Warm-up больше не нужен — understandQuery сам кеширует результаты
  // Эта функция оставлена для обратной совместимости
}

// Реэкспорт типов из queryUnderstanding для удобства
export type { QueryUnderstandingResult } from '../rag/queryUnderstanding';
