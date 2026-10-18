import { requestAppNavigation } from '../../navigation/AppNavigationBus';
import { GlobalSearchService, type GlobalSearchResult } from '../search/GlobalSearchService';
import { RagFacade } from '../rag/RagFacade';

const ROUTE_PATTERNS: Array<{ route: string; patterns: RegExp[] }> = [
  { route: 'dashboard', patterns: [/\b(dashboard|рабоч(ий|его)\s+стол|панел)/i] },
  { route: 'clients', patterns: [/\b(clients?|клиент|клиенты|проект|проекты|projects?)\b/i] },
  { route: 'users', patterns: [/\b(users?|staff|сотрудник|сотрудники|персонал)\b/i] },
  { route: 'knowledge', patterns: [/\b(knowledge|kb|knowledge base|база знаний)\b/i] },
  { route: 'laila', patterns: [/\b(laila|лайла|чат)\b/i] },
  { route: 'systematizer', patterns: [/\b(systematizer|систематизатор|систематизац)/i] },
  { route: 'settings', patterns: [/\b(settings?|настройк)/i] },
  { route: 'audit', patterns: [/\b(audit|журнал|логи?|аудит)\b/i] },
  { route: 'trash', patterns: [/\b(trash|корзин)/i] },
  { route: 'ai-control', patterns: [/\b(ai control|ollama|model(s)?|модел|оллама)\b/i] },
];

// [3.3] Regex навигация — без вызова LLM
const NAVIGATION_INTENT =
  /(open|go to|navigate to|show|take me to|switch to|find|search|перейди|открой|покажи|отведи|переключи|найди|отыщи)/i;

const COMMAND_PREFIX =
  /^\s*(open|go to|navigate to|show|take me to|switch to|find|search|перейди|открой|покажи|отведи|переключи|найди|отыщи)\s+/i;

export interface LailaActionResult {
  handled: boolean;
  route?: string;
  title?: string;
}

export interface ClientSearchResult {
  clientId: number;
  name: string;
  score: number;
}

function pickBestResult(input: string, results: GlobalSearchResult[]): GlobalSearchResult | null {
  if (!results.length) return null;
  const query = input.toLowerCase();
  return (
    results.find((item) => item.title.toLowerCase() === query) ??
    results.find((item) => item.title.toLowerCase().startsWith(query)) ??
    results[0]
  );
}

/**
 * [3.3] Навигация по regex без LLM.
 * Вызывается первым — быстро и без задержки.
 */
function tryRouteByPattern(input: string): LailaActionResult {
  const match = ROUTE_PATTERNS.find((entry) =>
    entry.patterns.some((pattern) => pattern.test(input)),
  );
  if (!match) return { handled: false };
  requestAppNavigation(match.route);
  return { handled: true, route: match.route };
}

export async function tryHandleLailaAction(input: string): Promise<LailaActionResult> {
  const trimmed = input.trim();
  if (!trimmed || !NAVIGATION_INTENT.test(trimmed)) {
    return { handled: false };
  }

  // [3.3] Сначала regex — без LLM
  const patternResult = tryRouteByPattern(trimmed);
  if (patternResult.handled) return patternResult;

  const normalizedQuery = trimmed.replace(COMMAND_PREFIX, '').trim();

  if (normalizedQuery) {
    const results = await GlobalSearchService.search(normalizedQuery, null);
    const best = pickBestResult(normalizedQuery, results);
    if (best) {
      requestAppNavigation(best.target.route, best.target);
      return { handled: true, route: best.target.route, title: best.title };
    }
  }

  return { handled: false };
}

/**
 * [3.1] Widget agent: поиск клиентов через RAG.
 * Используется виджетом чата для подсказок по клиентам.
 */
export async function searchClientsForWidget(query: string): Promise<ClientSearchResult[]> {
  if (!query.trim()) return [];
  try {
    const results = await RagFacade.searchClients(query, 5);
    return results
      .filter((r) => r.score > 0.3)
      .map((r) => ({
        clientId: Number(r.filePath.replace('virtual://clients/', '')),
        name: r.title,
        score: r.score,
      }))
      .filter((r) => Number.isFinite(r.clientId) && r.clientId > 0);
  } catch {
    return [];
  }
}
