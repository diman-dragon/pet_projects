// src/services/search/GhostRecordService.ts
//
// ИТЕРАЦИЯ 4: Проверка и очистка ghost/orphan записей в library.
//
// Ghost record — запись в library.file_path, которой нет на диске.
// Orphan — virtual:// запись, которой нет в clients_main.
//
// Запускается при старте приложения (после инициализации БД) и может
// вызываться вручную из аудит-панели.
//
// Поведение:
//   - orphaned = true  → помечает запись в metadata как orphaned
//   - удаление        → только если deleteOrphaned = true
//
// virtual:// сверяется с clients_main: clients, cases, lines, appeals.

import { exists } from '@tauri-apps/plugin-fs';
import { ConnectionManager } from '../db/ConnectionManager';
import { createLogger } from '../logger/Logger';

const log = createLogger('GhostRecordService');

export interface GhostCheckResult {
  checked: number;
  orphaned: number;
  deleted: number;
  errors: number;
}

function parseVirtualUri(filePath: string): {
  type: 'clients' | 'cases' | 'lines' | 'appeals' | null;
  id: number;
} | null {
  // virtual://clients/123, virtual://cases/123, etc.
  const m = filePath.match(/^virtual:\/\/(clients|cases|lines|appeals)\/(\d+)$/);
  if (!m) return null;
  return { type: m[1] as any, id: parseInt(m[2], 10) };
}

async function checkVirtualExists(filePath: string): Promise<boolean> {
  const parsed = parseVirtualUri(filePath);
  if (!parsed || !parsed.type) return true; // unknown virtual — не трогаем
  if (!ConnectionManager.exists('clients_main')) return true; // не можем проверить

  const db = ConnectionManager.get('clients_main');
  const tableMap: Record<string, string> = {
    clients: 'clients',
    cases: 'cases',
    lines: 'lines',
    appeals: 'appeals',
  };
  const table = tableMap[parsed.type];
  if (!table) return true;

  try {
    const rows = await db.select<Array<{ id: number }>>(
      `SELECT id FROM ${table} WHERE id = $1 AND deleted_at IS NULL LIMIT 1`,
      [parsed.id],
    );
    return rows.length > 0;
  } catch {
    return true; // ошибка — не трогаем
  }
}

async function markOrphaned(db: ReturnType<typeof ConnectionManager.get>, id: number): Promise<void> {
  try {
    // Читаем текущий metadata
    const rows = await db.select<Array<{ metadata: string | null }>>(
      `SELECT metadata FROM library WHERE id = $1`,
      [id],
    );
    const current = rows[0]?.metadata;
    let meta: Record<string, unknown> = {};
    try {
      meta = current ? JSON.parse(current) : {};
    } catch { /* */ }
    meta.orphaned = true;
    meta.orphaned_at = new Date().toISOString();
    await db.execute(
      `UPDATE library SET metadata = $1 WHERE id = $2`,
      [JSON.stringify(meta), id],
    );
  } catch (err) {
    log.warn(`Failed to mark orphaned id=${id}`, err);
  }
}

export const GhostRecordService = {
  /**
   * Основной метод: проверяет все записи library.
   *
   * @param deleteOrphaned  Если true — удаляет orphan-записи. По умолчанию false (только помечает).
   * @param signal          AbortSignal для прерывания долгой проверки.
   */
  async check(
    deleteOrphaned = false,
    signal?: AbortSignal,
  ): Promise<GhostCheckResult> {
    const result: GhostCheckResult = { checked: 0, orphaned: 0, deleted: 0, errors: 0 };

    if (!ConnectionManager.exists('kb_main')) return result;
    const db = ConnectionManager.get('kb_main');

    let rows: Array<{ id: number; file_path: string }>;
    try {
      rows = await db.select<Array<{ id: number; file_path: string }>>(
        `SELECT id, file_path FROM library ORDER BY id`,
      );
    } catch (err) {
      log.error('Failed to load library for ghost check', err);
      return result;
    }

    for (const row of rows) {
      if (signal?.aborted) break;

      result.checked++;
      const fp = row.file_path;

      try {
        let alive = true;

        if (fp.startsWith('virtual://')) {
          alive = await checkVirtualExists(fp);
        } else {
          alive = await exists(fp);
        }

        if (!alive) {
          result.orphaned++;
          if (deleteOrphaned) {
            await db.execute(`DELETE FROM library WHERE id = $1`, [row.id]);
            result.deleted++;
            log.info(`Deleted ghost record id=${row.id} path=${fp}`);
          } else {
            await markOrphaned(db, row.id);
            log.info(`Marked orphaned id=${row.id} path=${fp}`);
          }
        }
      } catch (err) {
        result.errors++;
        log.warn(`Error checking record id=${row.id}`, err);
      }
    }

    log.info('Ghost check complete', result);
    return result;
  },

  /**
   * Лёгкая версия: только считает ghost-записи без изменений.
   */
  async count(): Promise<number> {
    if (!ConnectionManager.exists('kb_main')) return 0;
    const db = ConnectionManager.get('kb_main');

    try {
      const rows = await db.select<Array<{ id: number; file_path: string }>>(
        `SELECT id, file_path FROM library WHERE file_path NOT LIKE 'virtual://%' ORDER BY id`,
      );
      let count = 0;
      for (const row of rows) {
        if (!(await exists(row.file_path))) count++;
      }
      return count;
    } catch {
      return 0;
    }
  },
};
