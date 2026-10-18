import { ConnectionManager } from '../db/ConnectionManager';
import { createLogger } from '../logger/Logger';

const log = createLogger('UnifiedSearchSupport');

export async function countRows(tableName: 'library' | 'chunks' | 'embeddings'): Promise<number> {
  const db = ConnectionManager.get('kb_main');
  const rows = (await db.select(`SELECT COUNT(*) AS cnt FROM ${tableName}`)) as
    | Array<{ cnt: number }>
    | undefined;
  return rows?.[0]?.cnt ?? 0;
}

export async function countEmbeddingsForModel(model: string): Promise<number> {
  const db = ConnectionManager.get('kb_main');
  const rows = (await db.select(`SELECT COUNT(*) AS cnt FROM embeddings WHERE model = $1`, [model])) as
    | Array<{ cnt: number }>
    | undefined;
  return rows?.[0]?.cnt ?? 0;
}

export async function getCompatibleEmbeddingModels(db: any, model: string): Promise<string[]> {
  try {
    const exactRows = (await db.select(`SELECT COUNT(*) AS cnt FROM embeddings WHERE model = $1 LIMIT 1`, [model])) as
      | Array<{ cnt: number }>
      | undefined;

    if ((exactRows?.[0]?.cnt ?? 0) > 0) {
      log.info(`[CompatibleModels] Exact match for: ${model}`);
      return [model];
    }

    const modelBase = model.split(':')[0];
    if (modelBase && modelBase !== model) {
      const baseRows = (await db.select(
        `SELECT DISTINCT model FROM embeddings WHERE model LIKE $1 || '%' LIMIT 5`,
        [modelBase],
      )) as Array<{ model: string }> | undefined;

      const found = (baseRows ?? []).map((row) => row.model).filter(Boolean);
      if (found.length > 0) {
        log.info(`[CompatibleModels] Base match for: ${model} -> ${found.join(', ')}`);
        return found;
      }
    }

    log.warn(`[CompatibleModels] No embeddings found for model: ${model}`);
    return [];
  } catch (error) {
    log.warn('[CompatibleModels] Error querying DB:', error);
    return [];
  }
}
