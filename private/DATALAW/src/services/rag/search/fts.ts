import type { SearchResult } from '../UnifiedSearchService';
import { isFtsOutOfSync, rebuildFtsIndex, runFtsQuery } from './ftsDb';
import { buildFtsExpression, buildSearchTerms, tokenizeQuery } from './ftsHelpers';
import { mapFtsRows, mapScanRows } from './ftsMappers';

export async function searchFts(
  db: any,
  query: string,
  limit: number,
  category: string | undefined,
  expandedTerms: string[] = [],
  queryLang?: string,
): Promise<SearchResult[]> {
  const tokens = tokenizeQuery(query);
  if (!tokens.length) {
    return searchByScan(db, query, limit, category, expandedTerms);
  }

  const ftsExpr = buildFtsExpression(tokens, expandedTerms);

  try {
    let rows = await runFtsQuery(db, ftsExpr, limit, category, queryLang);

    if (!rows.length && (await isFtsOutOfSync(db))) {
      await rebuildFtsIndex(db);
      rows = await runFtsQuery(db, ftsExpr, limit, category, queryLang);
    }

    if (!rows.length) {
      return searchByScan(db, query, limit, category, expandedTerms);
    }

    const maxScore = Math.max(...rows.map((row) => Number(row.bm25_score)));
    if (!Number.isFinite(maxScore) || maxScore <= 0) {
      return searchByScan(db, query, limit, category, expandedTerms);
    }

    return mapFtsRows(rows, maxScore);
  } catch {
    return searchByScan(db, query, limit, category, expandedTerms);
  }
}

async function searchByScan(
  db: any,
  query: string,
  limit: number,
  category: string | undefined,
  expandedTerms: string[] = [],
): Promise<SearchResult[]> {
  const { tokens, searchTerms } = buildSearchTerms(query, expandedTerms);
  if (!tokens.length) {
    return [];
  }

  const catParam = category && category !== 'all' ? category : null;

  try {
    const rows = (await db.select(
      `SELECT c.id AS chunk_id, c.library_id, c.chunk_index, c.text AS chunk_text,
              c.char_start, c.char_end, c.article_ref, c.parent_id,
              p.text AS parent_text,
              l.title, l.category, l.file_path, l.lang
       FROM chunks c
       JOIN library l ON l.id = c.library_id
       LEFT JOIN chunks p ON p.id = c.parent_id
       WHERE ($1 IS NULL OR l.category = $1)`,
      [catParam],
    )) as any[];

    return mapScanRows(rows, tokens, searchTerms).slice(0, limit);
  } catch {
    return [];
  }
}
