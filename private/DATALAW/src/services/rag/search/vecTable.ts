import { resetCapabilityCache } from './capability';

async function getCurrentVecDimension(db: any): Promise<number | null> {
  try {
    const rows = (await db.select(
      `SELECT sql FROM sqlite_master WHERE name = 'vec_embeddings' LIMIT 1`,
    )) as Array<{ sql: string | null }> | undefined;
    const sql = rows?.[0]?.sql ?? '';
    const match = sql.match(/FLOAT\[(\d+)\]/i);
    return match ? Number(match[1]) : null;
  } catch {
    return null;
  }
}

export async function ensureVecEmbeddingsTable(db: any, dim: number): Promise<void> {
  const currentDim = await getCurrentVecDimension(db);
  if (currentDim === dim) return;

  try {
    await db.execute(`DROP TABLE IF EXISTS vec_embeddings`);
  } catch {
    // ignore
  }

  await db.execute(
    `CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings
     USING vec0(
       chunk_id INTEGER PRIMARY KEY,
       library_id INTEGER,
       model TEXT,
       embedding FLOAT[${dim}]
     )`,
  );

  resetCapabilityCache();
}
