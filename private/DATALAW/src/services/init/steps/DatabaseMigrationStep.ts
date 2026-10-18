import { join } from '@tauri-apps/api/path';
import { exists, remove } from '@tauri-apps/plugin-fs';
import { USER_DB_SCHEMA } from '../../../constants/schemas/UserDbSchema';
import { KB_SCHEMA } from '../../../constants/schemas/KbSchema';
import { CLIENT_SCHEMA } from '../../../constants/schemas/ClientSchema';
import { PII_VAULT_SCHEMA } from '../../../constants/schemas/PiiVaultSchema';
import { LEGAL_CORPUS_SCHEMA } from '../../../constants/schemas/LegalCorpusSchema';
import { ConnectionManager } from '../../db/ConnectionManager';
import { MIGRATIONS } from '../../db/migrations';
import { createLogger } from '../../logger/Logger';
import type { FolderPaths, InitContext, InitStep } from '../types';

const log = createLogger('DatabaseMigration');

const SCHEMA_VERSION_DDL = `
  CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    db_key TEXT NOT NULL,
    version INTEGER NOT NULL,
    description TEXT,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(db_key, version)
  );
`;

const ALL_KEYS = ['user_system', 'kb_main', 'clients_main', 'pii_vault', 'legal_corpus'] as const;

const DATABASES = {
  user_system: {
    folderName: '.db_user',
    fileName: 'users.db',
    schema: USER_DB_SCHEMA,
    basePath: (paths: FolderPaths) => paths.rootPath,
  },
  kb_main: {
    folderName: '.db_kb',
    fileName: 'kb_index.db',
    schema: KB_SCHEMA,
    basePath: (paths: FolderPaths) => paths.kbPath,
  },
  clients_main: {
    folderName: '.db_clients',
    fileName: 'clients.db',
    schema: CLIENT_SCHEMA,
    basePath: (paths: FolderPaths) => paths.clientsPath,
  },
  pii_vault: {
    folderName: '.db_pii',
    fileName: 'pii_vault.db',
    schema: PII_VAULT_SCHEMA,
    basePath: (paths: FolderPaths) => paths.piiVaultPath,
  },
  legal_corpus: {
    folderName: '.db_legal',
    fileName: 'legal_corpus.db',
    schema: LEGAL_CORPUS_SCHEMA,
    basePath: (paths: FolderPaths) => paths.legalCorpusPath,
  },
} as const;

function isRecoverableMigrationError(error: unknown): boolean {
  const message = String((error as { message?: string })?.message ?? error ?? '').toLowerCase();
  return (
    message.includes('duplicate column name') ||
    message.includes('already exists') ||
    message.includes('no such module')
  );
}

export class DatabaseMigrationStep implements InitStep {
  readonly name = 'DatabaseMigration';

  private async rebuildDatabase(ctx: InitContext, key: (typeof ALL_KEYS)[number]): Promise<void> {
    const descriptor = DATABASES[key];
    const basePath = descriptor.basePath(ctx.paths!);
    const folderPath = await join(basePath, descriptor.folderName);
    const dbPath = await join(folderPath, descriptor.fileName);

    await ConnectionManager.close(key);
    if (await exists(dbPath)) {
      await remove(dbPath);
    }

    const db = await ConnectionManager.open(key, dbPath);
    await db.execute(descriptor.schema);
    await db.execute(SCHEMA_VERSION_DDL);
    log.warn(`База [${key}] была пересоздана с нуля из актуальной схемы.`);
  }

  async run(ctx: InitContext): Promise<InitContext> {
    for (const key of ALL_KEYS) {
      const db = ConnectionManager.get(key);
      await db.execute(SCHEMA_VERSION_DDL);
      log.info(`schema_version ready: [${key}]`);
    }

    const orderedMigrations = [...MIGRATIONS].sort(
      (left, right) => left.version - right.version || left.key.localeCompare(right.key),
    );

    for (const migration of orderedMigrations) {
      const db = ConnectionManager.get(migration.key);
      const rows = await db.select<{ count: number }[]>(
        `SELECT COUNT(*) as count
         FROM schema_version
         WHERE db_key = $1 AND version = $2`,
        [migration.key, migration.version],
      );

      if ((rows[0]?.count ?? 0) > 0) {
        continue;
      }

      log.info(`v${migration.version} [${migration.key}] ${migration.description}`);

      try {
        await db.execute(migration.sql);
      } catch (error) {
        if (!isRecoverableMigrationError(error)) {
          log.error(`Migration failed: v${migration.version} [${migration.key}]`, error);
          await this.rebuildDatabase(ctx, migration.key);

          const rebuiltDb = ConnectionManager.get(migration.key);
          const sameKeyMigrations = orderedMigrations.filter((item) => item.key === migration.key);
          for (const rebuiltMigration of sameKeyMigrations) {
            try {
              await rebuiltDb.execute(rebuiltMigration.sql);
            } catch (rebuiltError) {
              if (!isRecoverableMigrationError(rebuiltError)) {
                throw rebuiltError;
              }
            }

            await rebuiltDb.execute(
              `INSERT OR IGNORE INTO schema_version (db_key, version, description)
               VALUES ($1, $2, $3)`,
              [rebuiltMigration.key, rebuiltMigration.version, rebuiltMigration.description],
            );
          }

          continue;
        }

        log.warn(`Recovered migration: v${migration.version} [${migration.key}]`);
      }

      await db.execute(
        `INSERT OR IGNORE INTO schema_version (db_key, version, description)
         VALUES ($1, $2, $3)`,
        [migration.key, migration.version, migration.description],
      );
    }

    return ctx;
  }
}
