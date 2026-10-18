// src/services/search/GlobalSearchService.ts
//
// ИТЕРАЦИЯ 4: исправлен recordSelection (query_hash + корректная схема),
// логирование в search_history при каждом поиске,
// интеграция SearchCache для инвалидации при мутациях.
//
// Все изменения итерации 3 сохранены без изменений.

import { ConnectionManager } from '../db/ConnectionManager';
import { hasPermission } from '../auth/permissions';
import type { UserProfile } from '../../types/entities';
import { RagFacade } from '../rag/RagFacade';
import { processQuery, type ProcessedQuery } from './QueryProcessor';
import { SearchCache } from './SearchCache';
import { stemToken } from '../rag/utils/stemming';
import { shortHash } from '../utils/hash';

export type GlobalSearchTarget =
  | { route: 'clients'; clientId?: number; caseId?: number; lineId?: number; appealId?: number }
  | { route: 'knowledge' }
  | { route: 'users' };

export interface GlobalSearchResult {
  id: string;
  title: string;
  subtitle: string;
  section: 'clients' | 'cases' | 'lines' | 'appeals' | 'knowledge' | 'users';
  target: GlobalSearchTarget;
  score?: number;
  source?: 'fts' | 'db' | 'semantic' | 'exact';
  breadcrumb?: string;
  lang?: string;
}

// ── Утилиты ──────────────────────────────────────────────────────────────────

function canAccess(user: UserProfile | null | undefined, permission: string): boolean {
  return !user || hasPermission(user, permission as any);
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Timeout: ${label}`)), ms);
    promise.then(
      (v) => { clearTimeout(timer); resolve(v); },
      (e) => { clearTimeout(timer); reject(e); },
    );
  });
}

/**
 * RRF score: 1 / (k + rank), k=60
 */
function rrfScore(rank: number, k = 60): number {
  return 1 / (k + rank);
}

/**
 * Применяет RRF поверх нескольких упорядоченных списков результатов.
 */
function applyRRF(
  lists: GlobalSearchResult[][],
  queryLang: string,
  selectionPriorIds: Set<string>,
): GlobalSearchResult[] {
  const scoreMap = new Map<string, { result: GlobalSearchResult; rrf: number }>();

  for (const list of lists) {
    for (let i = 0; i < list.length; i++) {
      const r = list[i];
      const rrf = rrfScore(i + 1);
      const existing = scoreMap.get(r.id);
      if (existing) {
        existing.rrf += rrf;
      } else {
        scoreMap.set(r.id, { result: r, rrf });
      }
    }
  }

  return Array.from(scoreMap.values())
    .map(({ result, rrf }) => {
      let finalScore = rrf;

      // Language boost ×1.2
      if (result.lang && result.lang === queryLang) {
        finalScore *= 1.2;
      }

      // Search selection prior ×1.1
      if (selectionPriorIds.has(result.id)) {
        finalScore *= 1.1;
      }

      return { ...result, score: Math.round(finalScore * 10000) };
    })
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
}

/**
 * Строит FTS5 MATCH-выражение из токенов + expanded terms.
 */
function buildFtsExpr(pq: ProcessedQuery): string {
  if (pq.queryType === 'exact_identifier') {
    const escaped = pq.normalizedQuery.toLowerCase().replace(/"/g, '""');
    return `"${escaped}"`;
  }

  const terms = new Set<string>();

  for (const token of pq.tokens) {
    const lower = token.toLowerCase();
    terms.add(`"${lower.replace(/"/g, '""')}"*`);
    const stem = stemToken(token);
    if (stem && stem !== token) {
      terms.add(`"${stem.replace(/"/g, '""')}"*`);
    }
  }

  for (const term of pq.expandedTerms) {
    const lower = term.toLowerCase().replace(/"/g, '""');
    if (lower.length > 2) {
      terms.add(`"${lower}"*`);
    }
  }

  return Array.from(terms).join(' OR ');
}

const SOURCE_TIMEOUT_MS = 400;

// ── Exact-поиск для идентификаторов ──────────────────────────────────────────

async function searchExactClients(
  query: string,
): Promise<GlobalSearchResult[]> {
  if (!ConnectionManager.exists('clients_main')) return [];
  const db = ConnectionManager.get('clients_main');
  const term = `%${query.replace(/[%_]/g, (c) => `\\${c}`)}%`;

  const rows = await db.select<Array<{
    id: number; full_name: string; tax_id: string | null; status: string | null;
  }>>(
    `SELECT id, full_name, tax_id, status
     FROM clients
     WHERE (tax_id = $1 OR email = $1 OR full_name LIKE $2 ESCAPE '\\')
       AND deleted_at IS NULL
     LIMIT 5`,
    [query, term],
  );

  return rows.map((item) => ({
    id: `client-${item.id}`,
    title: item.full_name,
    subtitle: `Клиент · ИНН ${item.tax_id ?? '—'}`,
    section: 'clients' as const,
    target: { route: 'clients' as const, clientId: item.id },
    score: 100,
    source: 'exact' as const,
  }));
}

async function searchExactCases(
  query: string,
): Promise<GlobalSearchResult[]> {
  if (!ConnectionManager.exists('clients_main')) return [];
  const db = ConnectionManager.get('clients_main');

  const rows = await db.select<Array<{
    id: number; client_id: number; title: string; case_number: string | null;
  }>>(
    `SELECT id, client_id, title, case_number
     FROM cases
     WHERE case_number = $1
       AND deleted_at IS NULL
     LIMIT 5`,
    [query],
  );

  return rows.map((item) => ({
    id: `case-${item.id}`,
    title: item.title,
    subtitle: `Дело · ${item.case_number ?? '—'}`,
    section: 'cases' as const,
    target: { route: 'clients' as const, clientId: item.client_id, caseId: item.id },
    score: 100,
    source: 'exact' as const,
  }));
}

// ── FTS-источники ─────────────────────────────────────────────────────────────

async function searchLibrary(
  _pq: ProcessedQuery,
  ftsExpr: string,
  _signal?: AbortSignal,
): Promise<GlobalSearchResult[]> {
  if (!ConnectionManager.exists('kb_main')) return [];
  const db = ConnectionManager.get('kb_main');

  const rows = await db.select<Array<{
    id: number; title: string; category: string | null;
    folder_path: string | null; lang: string | null;
  }>>(
    `SELECT l.id, l.title, l.category, l.folder_path, l.lang
     FROM library_fts
     JOIN library l ON l.id = library_fts.rowid
     WHERE library_fts MATCH $1
       AND l.file_path NOT LIKE 'virtual://%'
     ORDER BY rank
     LIMIT 12`,
    [ftsExpr],
  );

  return rows.map((item) => {
    const folder = item.folder_path
      ? item.folder_path.split('/').filter(Boolean).slice(-1)[0]
      : null;
    const breadcrumb = [item.category ?? 'General', folder].filter(Boolean).join(' → ');
    return {
      id: `lib-${item.id}`,
      title: item.title,
      subtitle: `База знаний · ${breadcrumb}`,
      section: 'knowledge' as const,
      target: { route: 'knowledge' as const },
      score: 70,
      source: 'fts' as const,
      breadcrumb,
      lang: item.lang ?? undefined,
    };
  });
}

async function searchClients(
  _pq: ProcessedQuery,
  ftsExpr: string,
  _signal?: AbortSignal,
): Promise<GlobalSearchResult[]> {
  if (!ConnectionManager.exists('clients_main')) return [];
  const db = ConnectionManager.get('clients_main');

  const rows = await db.select<Array<{
    id: number; full_name: string; status: string | null;
  }>>(
    `SELECT c.id, c.full_name, c.status
     FROM clients_fts
     JOIN clients c ON c.id = clients_fts.rowid
     WHERE clients_fts MATCH $1
       AND c.deleted_at IS NULL
     ORDER BY rank
     LIMIT 8`,
    [ftsExpr],
  );

  return rows.map((item) => ({
    id: `client-${item.id}`,
    title: item.full_name,
    subtitle: `Клиент · ${item.status ?? 'без статуса'}`,
    section: 'clients' as const,
    target: { route: 'clients' as const, clientId: item.id },
    score: 80,
    source: 'fts' as const,
  }));
}

async function searchCases(
  _pq: ProcessedQuery,
  ftsExpr: string,
  _signal?: AbortSignal,
): Promise<GlobalSearchResult[]> {
  if (!ConnectionManager.exists('clients_main')) return [];
  const db = ConnectionManager.get('clients_main');

  const rows = await db.select<Array<{
    id: number; client_id: number; title: string; case_type: string | null;
  }>>(
    `SELECT cs.id, cs.client_id, cs.title, cs.case_type
     FROM cases_fts
     JOIN cases cs ON cs.id = cases_fts.rowid
     WHERE cases_fts MATCH $1
       AND cs.deleted_at IS NULL
     ORDER BY rank
     LIMIT 8`,
    [ftsExpr],
  );

  return rows.map((item) => ({
    id: `case-${item.id}`,
    title: item.title,
    subtitle: `Дело · ${item.case_type ?? 'без типа'}`,
    section: 'cases' as const,
    target: { route: 'clients' as const, clientId: item.client_id, caseId: item.id },
    score: 75,
    source: 'fts' as const,
  }));
}

async function searchAppeals(
  _pq: ProcessedQuery,
  ftsExpr: string,
  _signal?: AbortSignal,
): Promise<GlobalSearchResult[]> {
  if (!ConnectionManager.exists('clients_main')) return [];
  const db = ConnectionManager.get('clients_main');

  const rows = await db.select<Array<{
    id: number; line_id: number; title: string; status: string | null;
    case_id: number; client_id: number;
  }>>(
    `SELECT a.id, a.line_id, a.title, a.status, l.case_id, c.client_id
     FROM appeals_fts
     JOIN appeals a ON a.id = appeals_fts.rowid
     JOIN lines l ON l.id = a.line_id
     JOIN cases c ON c.id = l.case_id
     WHERE appeals_fts MATCH $1
       AND a.deleted_at IS NULL
     ORDER BY rank
     LIMIT 8`,
    [ftsExpr],
  );

  return rows.map((item) => ({
    id: `appeal-${item.id}`,
    title: item.title,
    subtitle: `Обращение · ${item.status ?? 'без статуса'}`,
    section: 'appeals' as const,
    target: {
      route: 'clients' as const,
      clientId: item.client_id,
      caseId: item.case_id,
      lineId: item.line_id,
      appealId: item.id,
    },
    score: 70,
    source: 'fts' as const,
  }));
}

async function searchUsers(
  pq: ProcessedQuery,
  _signal?: AbortSignal,
): Promise<GlobalSearchResult[]> {
  if (!ConnectionManager.exists('user_system')) return [];
  const db = ConnectionManager.get('user_system');
  const term = `%${pq.normalizedQuery.replace(/[%_]/g, (c) => `\\${c}`)}%`;

  const rows = await db.select<Array<{
    id: number; full_name: string; login: string; role: string;
  }>>(
    `SELECT id, full_name, login, role
     FROM users
     WHERE full_name LIKE $1 ESCAPE '\\' OR login LIKE $1 ESCAPE '\\'
     ORDER BY full_name ASC
     LIMIT 5`,
    [term],
  );

  return rows.map((item) => ({
    id: `user-${item.id}`,
    title: item.full_name,
    subtitle: `Сотрудник · @${item.login} · ${item.role}`,
    section: 'users' as const,
    target: { route: 'users' as const },
    score: 60,
    source: 'db' as const,
  }));
}

async function searchSemantic(query: string): Promise<GlobalSearchResult[]> {
  try {
    const ragResults = await RagFacade.searchAll(query, 5);
    return ragResults.map((r) => ({
      id: `semantic-${r.chunkId}`,
      title: r.title,
      subtitle: `Семантика · ${r.category ?? 'База знаний'} · ${r.score.toFixed(2)}`,
      section: 'knowledge' as const,
      target: { route: 'knowledge' as const },
      score: Math.round(r.score * 100),
      source: 'semantic' as const,
    }));
  } catch {
    return [];
  }
}

/**
 * Загружает id результатов, которые пользователь выбирал ранее.
 */
async function loadSelectionPriors(): Promise<Set<string>> {
  const ids = new Set<string>();
  if (!ConnectionManager.exists('kb_main')) return ids;
  try {
    const db = ConnectionManager.get('kb_main');
    const rows = await db.select<Array<{ result_id: string }>>(
      `SELECT result_id FROM search_selections ORDER BY selected_at DESC LIMIT 50`,
    );
    for (const r of rows) ids.add(r.result_id);
  } catch {
    // таблицы нет — игнорируем
  }
  return ids;
}

/**
 * Логирует запрос в search_history для suggestions и аналитики.
 */
async function logSearchHistory(
  query: string,
  queryLang: string,
  queryType: string,
  latencyMs: number,
): Promise<void> {
  if (!ConnectionManager.exists('kb_main')) return;
  try {
    const db = ConnectionManager.get('kb_main');
    const qHash = shortHash(query);
    await db.execute(
      `INSERT INTO search_history (query, query_hash, query_lang, query_type, latency_ms, created_at)
       VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)`,
      [query, qHash, queryLang, queryType, latencyMs],
    );
  } catch {
    // некритично — таблица могла не быть создана на старых версиях
  }
}

// ── Публичный API ─────────────────────────────────────────────────────────────

export const GlobalSearchService = {
  /**
   * Записывает выбор результата в search_selections (prior для будущих поисков).
   * ИСПРАВЛЕНИЕ итерации 4: корректные поля согласно схеме миграции 208.
   */
  async recordSelection(resultId: string, queryHash?: string): Promise<void> {
    if (!ConnectionManager.exists('kb_main')) return;
    try {
      const db = ConnectionManager.get('kb_main');
      // Извлекаем library_id если это lib-N
      const libMatch = resultId.match(/^lib-(\d+)$/);
      const libraryId = libMatch ? parseInt(libMatch[1], 10) : null;

      await db.execute(
        `INSERT INTO search_selections (query_hash, library_id, result_id, selected_at)
         VALUES ($1, $2, $3, CURRENT_TIMESTAMP)`,
        [queryHash ?? '', libraryId, resultId],
      );

      // Инвалидируем кеш т.к. selection prior изменился
      SearchCache.invalidateOnMutation();
    } catch {
      // некритично
    }
  },

  async search(
    query: string,
    user?: UserProfile | null,
    signal?: AbortSignal,
  ): Promise<GlobalSearchResult[]> {
    const trimmed = query.trim();
    if (!trimmed) return [];

    const t0 = Date.now();

    // ── QueryProcessor ──────────────────────────────────────────────────────
    let pq: ProcessedQuery;
    try {
      pq = await processQuery(trimmed);
    } catch {
      pq = {
        normalizedQuery: trimmed,
        tokens: trimmed.toLowerCase().split(/\s+/).filter((t) => t.length > 1),
        queryType: 'free_text',
        queryLang: 'ru',
        expandedTerms: [],
      };
    }

    if (!pq.tokens.length && pq.queryType !== 'exact_identifier') return [];

    // ── Exact-поиск для идентификаторов ────────────────────────────────────
    if (pq.queryType === 'exact_identifier') {
      const tasks: Promise<GlobalSearchResult[]>[] = [];

      if (canAccess(user, 'clients:view')) {
        tasks.push(
          withTimeout(searchExactClients(pq.normalizedQuery), SOURCE_TIMEOUT_MS, 'exact_clients'),
          withTimeout(searchExactCases(pq.normalizedQuery), SOURCE_TIMEOUT_MS, 'exact_cases'),
        );
      }

      const settled = await Promise.allSettled(tasks);
      const results: GlobalSearchResult[] = [];
      const seen = new Set<string>();

      for (const o of settled) {
        if (o.status === 'fulfilled') {
          for (const r of o.value) {
            if (!seen.has(r.id)) { seen.add(r.id); results.push(r); }
          }
        }
      }

      const final = results.sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).slice(0, 15);

      // Логируем в history
      logSearchHistory(trimmed, pq.queryLang, pq.queryType, Date.now() - t0).catch(() => { });

      return final;
    }

    // ── FTS-выражение ───────────────────────────────────────────────────────
    const ftsExpr = buildFtsExpr(pq);
    if (!ftsExpr) return [];

    // ── Параллельный запрос источников ─────────────────────────────────────
    const sourceTasks: Array<[string, Promise<GlobalSearchResult[]>]> = [];

    if (canAccess(user, 'knowledge:view')) {
      sourceTasks.push([
        'library',
        withTimeout(searchLibrary(pq, ftsExpr, signal), SOURCE_TIMEOUT_MS, 'library'),
      ]);
    }

    if (canAccess(user, 'clients:view')) {
      sourceTasks.push(
        ['clients', withTimeout(searchClients(pq, ftsExpr, signal), SOURCE_TIMEOUT_MS, 'clients')],
        ['cases', withTimeout(searchCases(pq, ftsExpr, signal), SOURCE_TIMEOUT_MS, 'cases')],
        ['appeals', withTimeout(searchAppeals(pq, ftsExpr, signal), SOURCE_TIMEOUT_MS, 'appeals')],
      );
    }

    if (canAccess(user, 'users:view')) {
      sourceTasks.push([
        'users',
        withTimeout(searchUsers(pq, signal), SOURCE_TIMEOUT_MS, 'users'),
      ]);
    }

    const [settled, selectionPriors] = await Promise.all([
      Promise.allSettled(sourceTasks.map(([, p]) => p)),
      loadSelectionPriors(),
    ]);

    const sourceLists: GlobalSearchResult[][] = [];
    for (const outcome of settled) {
      if (outcome.status === 'fulfilled' && outcome.value.length > 0) {
        sourceLists.push(outcome.value);
      }
    }

    // ── RRF merge ──────────────────────────────────────────────────────────
    const merged = applyRRF(sourceLists, pq.queryLang, selectionPriors);

    // ── Семантика (необязательная вторая фаза) ─────────────────────────────
    if (canAccess(user, 'knowledge:view') && !signal?.aborted) {
      try {
        const semantic = await withTimeout(
          searchSemantic(pq.normalizedQuery), SOURCE_TIMEOUT_MS, 'semantic',
        );
        const seen = new Set(merged.map((r) => r.id));
        for (const r of semantic) {
          if (!seen.has(r.id)) {
            seen.add(r.id);
            merged.push(r);
          }
        }
      } catch {
        // RAG недоступен — молча пропускаем
      }
    }

    const final = merged.slice(0, 25);

    // Логируем в search_history (фоново)
    logSearchHistory(trimmed, pq.queryLang, pq.queryType, Date.now() - t0).catch(() => { });

    return final;
  },
};
