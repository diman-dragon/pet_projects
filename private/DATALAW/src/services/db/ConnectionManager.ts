import { invoke } from '@tauri-apps/api/core';
import { createLogger } from '../logger/Logger';
import { resetCapabilityCache } from '../rag/search/capability';
import { MasterKeyService } from '../security/MasterKeyService';

interface DatabaseHandle {
  key: string;
  dbPath: string;
}

const connections = new Map<string, DatabaseHandle>();
const log = createLogger('ConnectionManager');

export interface DatabaseLike {
  path: string;
  close(): Promise<void>;
  select<T = unknown>(sql: string, params?: unknown[]): Promise<T>;
  execute(sql: string, params?: unknown[]): Promise<{ changes: number; lastInsertId: number | bigint | null }>;
}

function createDatabaseHandle(dbPath: string): DatabaseLike {
  return {
    path: dbPath,
    async close(): Promise<void> {
      return undefined;
    },
    async select<T = unknown>(sql: string, params: unknown[] = []): Promise<T> {
      const rows = await invoke<{ rows: Array<Record<string, unknown>> }>('db_select', {
        request: {
          dbPath,
          sql,
          params,
          keyHex: MasterKeyService.getSqlCipherKey() ?? undefined,
        },
      });
      return rows.rows as T;
    },
    async execute(sql: string, params: unknown[] = []): Promise<{ changes: number; lastInsertId: number | bigint | null }> {
      const result = await invoke<{ changed: number; lastInsertId: number | bigint | null }>('db_execute', {
        request: {
          dbPath,
          sql,
          params,
          keyHex: MasterKeyService.getSqlCipherKey() ?? undefined,
        },
      });
      return { changes: result.changed, lastInsertId: result.lastInsertId };
    },
  };
}

export const ConnectionManager = {
  async open(key: string, dbPath: string): Promise<DatabaseLike> {
    const existing = connections.get(key);
    if (existing) return createDatabaseHandle(existing.dbPath);

    if ((key === 'clients_main' || key === 'pii_vault') && !MasterKeyService.isUnlocked()) {
      throw new Error(`[ConnectionManager] ${key} requires an unlocked master key`);
    }

    await invoke('db_initialize', {
      request: {
        dbPath,
        sql: '',
        params: [],
        keyHex: MasterKeyService.getSqlCipherKey() ?? undefined,
      },
    });

    const handle = createDatabaseHandle(dbPath);
    connections.set(key, { key, dbPath });
    resetCapabilityCache();
    log.info(`Opened [${key}]`, dbPath);
    return handle;
  },

  exists(key: string): boolean {
    return connections.has(key);
  },

  get(key: string): DatabaseLike {
    const db = connections.get(key);
    if (!db) {
      throw new Error(`[ConnectionManager] Connection not found: ${key}`);
    }
    return createDatabaseHandle(db.dbPath);
  },

  async close(key: string): Promise<void> {
    const db = connections.get(key);
    if (db) {
      await invoke('db_close', {
        request: {
          dbPath: db.dbPath,
          sql: '',
          params: [],
          keyHex: MasterKeyService.getSqlCipherKey() ?? undefined,
        },
      }).catch(() => {});
    }
    connections.delete(key);
    log.info(`Closed [${key}]`);
  },

  async closeAll(): Promise<void> {
    for (const key of [...connections.keys()]) {
      await this.close(key);
    }
  },
};
