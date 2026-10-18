// src/services/init/steps/DatabaseSchemaStep.ts
// Ответственность: применить DDL-схемы к уже открытым соединениям.
// Требует, чтобы DatabaseConnectStep уже отработал.
// Схемы должны быть написаны с CREATE TABLE IF NOT EXISTS — идемпотентно.

import { ConnectionManager } from '../../db/ConnectionManager';
import { USER_DB_SCHEMA } from '../../../constants/schemas/UserDbSchema';
import { KB_SCHEMA } from '../../../constants/schemas/KbSchema';
import { CLIENT_SCHEMA } from '../../../constants/schemas/ClientSchema';
import { PII_VAULT_SCHEMA } from '../../../constants/schemas/PiiVaultSchema';
import { LEGAL_CORPUS_SCHEMA } from '../../../constants/schemas/LegalCorpusSchema';
import { createLogger, registerLogDbHandler } from '../../logger/Logger';
import { InitStep, InitContext } from '../types';

const log = createLogger('DatabaseSchema');

interface SchemaDescriptor {
  key: string;
  schema: string;
}

const SCHEMAS: SchemaDescriptor[] = [
  { key: 'user_system', schema: USER_DB_SCHEMA },
  { key: 'kb_main', schema: KB_SCHEMA },
  { key: 'clients_main', schema: CLIENT_SCHEMA },
  { key: 'pii_vault', schema: PII_VAULT_SCHEMA },
  { key: 'legal_corpus', schema: LEGAL_CORPUS_SCHEMA },
];

export class DatabaseSchemaStep implements InitStep {
  readonly name = 'DatabaseSchema';

  async run(ctx: InitContext): Promise<InitContext> {
    for (const { key, schema } of SCHEMAS) {
      const db = ConnectionManager.get(key);
      await db.execute(schema);
      log.info(`Схема применена: [${key}]`);
    }

    // Регистрация БД-логгера после того, как таблицы созданы
    registerLogDbHandler(async (entry) => {
      if (!ConnectionManager.exists('user_system')) return;
      const db = ConnectionManager.get('user_system');
      await db.execute(
        `INSERT INTO app_logs (level, tag, message, details, created_at) VALUES ($1, $2, $3, $4, $5)`,
        [entry.level, entry.tag, entry.message, entry.details ?? null, entry.timestamp],
      ).catch(() => {}); // Игнорируем ошибки записи лога, чтобы не упасть
    });

    return ctx;
  }
}
