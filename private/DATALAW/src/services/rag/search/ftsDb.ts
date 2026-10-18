export async function isFtsOutOfSync(db: any): Promise<boolean> {
  try {
    const rows = (await db.select(
      `SELECT
         (SELECT COUNT(*) FROM chunks) AS chunks_count,
         (SELECT COUNT(*) FROM chunks_fts) AS fts_count`,
    )) as Array<{ chunks_count: number; fts_count: number }> | undefined;

    const chunksCount = rows?.[0]?.chunks_count ?? 0;
    const ftsCount = rows?.[0]?.fts_count ?? 0;
    return chunksCount > 0 && ftsCount < chunksCount;
  } catch {
    return false;
  }
}

export async function rebuildFtsIndex(db: any): Promise<void> {
  await db.execute(`INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')`);
}

export async function runFtsQuery(
  db: any,
  ftsExpr: string,
  limit: number,
  category: string | undefined,
  queryLang?: string,
): Promise<any[]> {
  const catParam = category && category !== 'all' ? category : null;
  const rows = ((await db.select(
    `SELECT c.id AS chunk_id, c.library_id, c.chunk_index, c.text AS chunk_text,
            c.char_start, c.char_end, c.article_ref, c.parent_id,
            p.text AS parent_text,
            l.title, l.category, l.file_path, l.lang,
            (-bm25(chunks_fts, 3.0, 1.0)) AS bm25_score_raw
     FROM chunks_fts
     JOIN chunks c ON c.id = chunks_fts.rowid
     JOIN library l ON l.id = c.library_id
     LEFT JOIN chunks p ON p.id = c.parent_id
     WHERE chunks_fts MATCH $1
       AND ($3 IS NULL OR l.category = $3)
     ORDER BY bm25_score_raw DESC
     LIMIT $2`,
    [ftsExpr, limit, catParam],
  )) as any[]) ?? [];

  return rows.map((row) => ({
    ...row,
    bm25_score: Number(row.bm25_score_raw) * (queryLang && row.lang === queryLang ? 1.2 : 1.0),
  }));
}
