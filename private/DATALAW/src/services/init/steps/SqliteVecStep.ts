// src/services/init/steps/SqliteVecStep.ts
//
// Шаг пайплайна: проверяет что sqlite-vec зарегистрирован (Rust auto_extension),
// сбрасывает кеш capability чтобы detectCapability() переопределил режим поиска
// после того как соединения открыты.
//
// Порядок: DatabaseConnect → SqliteVecStep → DatabaseSchema → DatabaseMigration

import { invoke } from '@tauri-apps/api/core';
import { ConnectionManager } from '../../db/ConnectionManager';
import { resetCapabilityCache } from '../../rag/search/capability';
import { createLogger } from '../../logger/Logger';
import type { InitContext, InitStep } from '../types';

const log = createLogger('SqliteVecStep');

export class SqliteVecStep implements InitStep {
  readonly name = 'SqliteVecStep';

  async run(ctx: InitContext): Promise<InitContext> {
    // 1. Подтверждаем что Rust-сторона зарегистрировала расширение
    try {
      const msg = await invoke<string>('check_sqlite_vec');
      log.info(`Rust side: ${msg}`);
    } catch (e) {
      // Команда недоступна (dev-режим без Tauri) — не критично
      log.warn('check_sqlite_vec invoke failed (non-Tauri env?)', e);
    }

    // 2. Проверяем что vec_version() доступна в уже открытом соединении kb_main
    try {
      const db = ConnectionManager.get('kb_main');
      const rows = await db.select<{ v: string }[]>(`SELECT vec_version() AS v`);
      const version = rows?.[0]?.v ?? 'unknown';
      log.info(`sqlite-vec ready: ${version}`);
    } catch (e) {
      // Если даже vec_version() недоступна — расширение не загрузилось.
      // Это не фатально: поиск упадёт на js_fallback через detectCapability().
      log.warn('sqlite-vec NOT available in kb_main — fallback to js cosine', e);
    }

    // 3. Сброс кеша: соединения только что открыты, нужен свежий detekt
    resetCapabilityCache();

    return ctx;
  }
}
