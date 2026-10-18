// src/services/init/steps/DatabaseConnectStep.ts
// Ответственность: открыть SQLite-соединения для всех трёх баз.
// Не применяет схемы — это задача DatabaseSchemaStep.
// Идемпотентно: ConnectionManager пропускает уже открытые соединения.

import { join } from '@tauri-apps/api/path';
import { mkdir, exists } from '@tauri-apps/plugin-fs';
import { ConnectionManager } from '../../db/ConnectionManager';
import { createLogger } from '../../logger/Logger';
import { InitStep, InitContext, FolderPaths } from '../types';

const log = createLogger('DatabaseConnect');

interface DbDescriptor {
  folderName: string;
  fileName: string;
  key: string;
  basePath: (paths: FolderPaths) => string;
}

const DATABASES: DbDescriptor[] = [
  { folderName: '.db_user', fileName: 'users.db', key: 'user_system', basePath: (p) => p.rootPath },
  { folderName: '.db_kb', fileName: 'kb_index.db', key: 'kb_main', basePath: (p) => p.kbPath },
  {
    folderName: '.db_clients',
    fileName: 'clients.db',
    key: 'clients_main',
    basePath: (p) => p.clientsPath,
  },
  {
    folderName: '.db_pii',
    fileName: 'pii_vault.db',
    key: 'pii_vault',
    basePath: (p) => p.piiVaultPath,
  },
  {
    folderName: '.db_legal',
    fileName: 'legal_corpus.db',
    key: 'legal_corpus',
    basePath: (p) => p.legalCorpusPath,
  },
];

export class DatabaseConnectStep implements InitStep {
  readonly name = 'DatabaseConnect';

  async run(ctx: InitContext): Promise<InitContext> {
    const { paths } = ctx;

    for (const db of DATABASES) {
      const base = db.basePath(paths!);
      const folderPath = await join(base, db.folderName);

      if (!(await exists(folderPath))) {
        await mkdir(folderPath, { recursive: true });
      }

      const dbPath = await join(folderPath, db.fileName);
      await ConnectionManager.open(db.key, dbPath);
      log.info(`Соединение открыто: [${db.key}]`);
    }

    return ctx;
  }
}
