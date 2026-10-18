import { ConnectionManager } from '../db/ConnectionManager';
import { createLogger } from '../logger/Logger';
import { DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS } from './embeddingModels';

const log = createLogger('EmbeddingModelService');
const SETTING_KEY = 'rag_embedding_model';

let cachedModel: string | null = null;

export const EmbeddingModelService = {
  async get(): Promise<string> {
    if (cachedModel) return cachedModel;

    try {
      if (!ConnectionManager.exists('user_system')) {
        cachedModel = DEFAULT_EMBEDDING_MODEL;
        return cachedModel;
      }

      const db = ConnectionManager.get('user_system');
      const rows = (await db.select(
        `SELECT value FROM app_settings WHERE key = $1 LIMIT 1`,
        [SETTING_KEY],
      )) as Array<{ value: string }> | undefined;

      const stored = rows?.[0]?.value?.trim();
      cachedModel =
        stored && EMBEDDING_MODELS.some((item) => item.id === stored)
          ? stored
          : DEFAULT_EMBEDDING_MODEL;
      return cachedModel;
    } catch (error) {
      log.warn('Failed to read embedding model setting, using default', error);
      cachedModel = DEFAULT_EMBEDDING_MODEL;
      return cachedModel;
    }
  },

  async set(model: string): Promise<void> {
    cachedModel = model;

    try {
      if (!ConnectionManager.exists('user_system')) return;

      const db = ConnectionManager.get('user_system');
      await db.execute(
        `INSERT INTO app_settings (key, value, updated_at)
         VALUES ($1, $2, CURRENT_TIMESTAMP)
         ON CONFLICT(key) DO UPDATE SET value = $2, updated_at = CURRENT_TIMESTAMP`,
        [SETTING_KEY, model],
      );
    } catch (error) {
      log.error('Failed to store embedding model setting', error);
    }
  },

  list() {
    return EMBEDDING_MODELS;
  },

  getDefault(): string {
    return DEFAULT_EMBEDDING_MODEL;
  },

  clearCache(): void {
    cachedModel = null;
  },
};
