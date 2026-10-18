import type { SearchResult } from '../UnifiedSearchService';
import { calculateScanScore } from './ftsHelpers';

export function mapFtsRows(rows: any[], maxScore: number): SearchResult[] {
  return rows.map((row) => ({
    chunkId: row.chunk_id as number,
    libraryId: row.library_id as number,
    title: row.title as string,
    category: row.category as string,
    filePath: row.file_path as string,
    chunkText: row.chunk_text as string,
    chunkIndex: row.chunk_index as number,
    score: Math.min(Number(row.bm25_score) / maxScore, 1),
    matchType: 'text' as const,
    lang: row.lang as string | undefined,
    charStart: row.char_start as number | undefined,
    charEnd: row.char_end as number | undefined,
    articleRef: row.article_ref as string | null | undefined,
    parentId: row.parent_id as number | undefined,
    parentText: row.parent_text as string | undefined,
  }));
}

export function mapScanRows(rows: any[], tokens: string[], searchTerms: string[]): SearchResult[] {
  return (rows ?? [])
    .map((row) => {
      const title = String(row.title ?? '');
      const chunkText = String(row.chunk_text ?? '');
      const score = calculateScanScore(title, chunkText, tokens, searchTerms);

      return {
        chunkId: row.chunk_id as number,
        libraryId: row.library_id as number,
        title,
        category: row.category as string,
        filePath: row.file_path as string,
        chunkText,
        chunkIndex: row.chunk_index as number,
        score,
        matchType: 'text' as const,
        lang: row.lang as string | undefined,
        charStart: row.char_start as number | undefined,
        charEnd: row.char_end as number | undefined,
        articleRef: row.article_ref as string | null | undefined,
        parentId: row.parent_id as number | undefined,
        parentText: row.parent_text as string | undefined,
      };
    })
    .filter((row) => row.score >= 0.2)
    .sort((left, right) => right.score - left.score);
}
