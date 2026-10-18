import { createLogger } from '../../logger/Logger';

export type DbCapability = 'vec0' | 'vec_scalar' | 'js_fallback';

const cache = new Map<string, DbCapability>();
const log = createLogger('SqliteVecCapability');

export async function detectCapability(db: any, dbKey = 'kb_main'): Promise<DbCapability> {
  if (cache.has(dbKey)) return cache.get(dbKey)!;

  let cap: DbCapability;

  try {
    await db.select(`SELECT vec_version()`);
    try {
      await db.select(`SELECT count(*) FROM vec_embeddings LIMIT 1`);
      cap = 'vec0';
      log.info('Mode: vec0');
    } catch {
      cap = 'vec_scalar';
      log.info('Mode: vec_scalar');
    }
  } catch {
    cap = 'js_fallback';
    log.warn('Mode: js_fallback');
  }

  cache.set(dbKey, cap);
  return cap;
}

export function resetCapabilityCache(): void {
  cache.clear();
}
