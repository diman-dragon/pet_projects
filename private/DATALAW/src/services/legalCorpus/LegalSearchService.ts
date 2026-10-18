import { ConnectionManager } from '../db/ConnectionManager';
import type { LegalSearchResult } from './types';

export interface LegalSearchOptions {
  jurisdictionId?: number | null;
  limit?: number;
}

export const LegalSearchService = {
  async searchText(query: string, options: LegalSearchOptions = {}): Promise<LegalSearchResult[]> {
    const db = ConnectionManager.get('legal_corpus');
    const limit = options.limit ?? 10;
    const jurisdictionFilter = options.jurisdictionId ? 'AND acts.jurisdiction_id = $2' : '';
    const params = options.jurisdictionId ? [query, options.jurisdictionId, limit] : [query, limit];
    const limitParam = options.jurisdictionId ? '$3' : '$2';

    return db.select<LegalSearchResult[]>(
      `SELECT
         act_chunks.id AS chunkId,
         acts.id AS actId,
         acts.title AS title,
         act_chunks.article_ref AS articleRef,
         act_chunks.text AS text,
         bm25(act_chunks_fts) AS rank,
         acts.jurisdiction_id AS jurisdictionId
       FROM act_chunks_fts
       JOIN act_chunks ON act_chunks.id = act_chunks_fts.rowid
       JOIN acts ON acts.id = act_chunks.act_id
       WHERE act_chunks_fts MATCH $1
       ${jurisdictionFilter}
       ORDER BY rank
       LIMIT ${limitParam}`,
      params,
    );
  },

  async getChunksByIds(chunkIds: number[]): Promise<LegalSearchResult[]> {
    if (chunkIds.length === 0) return [];
    const db = ConnectionManager.get('legal_corpus');
    const results: LegalSearchResult[] = [];
    for (const id of chunkIds) {
      const [row] = await db.select<LegalSearchResult[]>(
        `SELECT
           act_chunks.id AS chunkId,
           acts.id AS actId,
           acts.title AS title,
           act_chunks.article_ref AS articleRef,
           act_chunks.text AS text,
           0 AS rank,
           acts.jurisdiction_id AS jurisdictionId
         FROM act_chunks
         JOIN acts ON acts.id = act_chunks.act_id
         WHERE act_chunks.id = $1`,
        [id],
      );
      if (row) results.push(row);
    }
    return results;
  },
};
