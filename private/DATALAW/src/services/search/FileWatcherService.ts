// src/services/search/FileWatcherService.ts
//
// ИТЕРАЦИЯ 4: File watcher с debounce для автоматической переиндексации.
//
// Поведение:
//   - Следит за папкой базы знаний через Tauri watch API
//   - Debounce 1500ms (минимум по плану) перед вызовом DocumentIndexer
//   - Создание/изменение файла → DocumentIndexer.index (быстрый режим)
//   - Удаление файла           → помечает запись orphaned в library
//   - Перемещение/переименование папки → recompute category для содержимого
//     (не трогает записи с category_overridden = true)
//
// Использование:
//   await FileWatcherService.start(kbRootPath);
//   FileWatcherService.stop();

import { watch } from '@tauri-apps/plugin-fs';
import type { UnlistenFn } from '@tauri-apps/api/event';
import { ConnectionManager } from '../db/ConnectionManager';
import { DocumentIndexer } from './DocumentIndexer';
import { FolderCategorizerService } from './FolderCategorizerService';
import { createLogger } from '../logger/Logger';
import { SearchCache } from './SearchCache';
import { invalidateSuggestionsCache } from './SearchSuggestionsService';

const log = createLogger('FileWatcherService');

const DEBOUNCE_MS = 1_500;
const SUPPORTED_EXTENSIONS = new Set([
  'pdf', 'docx', 'doc', 'txt', 'md', 'tsv', 'csv',
]);

function isSupportedFile(path: string): boolean {
  const ext = path.split('.').pop()?.toLowerCase() ?? '';
  return SUPPORTED_EXTENSIONS.has(ext);
}

function getExtension(path: string): string {
  return path.split('.').pop()?.toLowerCase() ?? '';
}

// Накапливаем события, чтобы батчевать debounce
const pending = new Map<string, { kind: 'create' | 'modify' | 'remove' | 'rename'; timer: ReturnType<typeof setTimeout> }>();

let unlisten: UnlistenFn | null = null;
let isRunning = false;

async function handleFileCreate(path: string): Promise<void> {
  if (!isSupportedFile(path)) return;
  log.info('File created/modified, indexing', path);
  try {
    await DocumentIndexer.index({ filePath: path, skipEmbeddings: false });
    SearchCache.invalidateOnMutation();
    invalidateSuggestionsCache();
  } catch (err) {
    log.warn('Failed to index on watch event', { path, err });
  }
}

async function handleFileRemove(path: string): Promise<void> {
  if (!ConnectionManager.exists('kb_main')) return;
  log.info('File removed, marking orphaned', path);
  const db = ConnectionManager.get('kb_main');
  try {
    const rows = await db.select<Array<{ id: number; metadata: string | null }>>(
      `SELECT id, metadata FROM library WHERE file_path = $1`,
      [path],
    );
    for (const row of rows) {
      let meta: Record<string, unknown> = {};
      try { meta = row.metadata ? JSON.parse(row.metadata) : {}; } catch { /* */ }
      meta.orphaned = true;
      meta.orphaned_at = new Date().toISOString();
      await db.execute(
        `UPDATE library SET metadata = $1 WHERE id = $2`,
        [JSON.stringify(meta), row.id],
      );
    }
    if (rows.length > 0) {
      SearchCache.invalidateOnMutation();
      invalidateSuggestionsCache();
    }
  } catch (err) {
    log.warn('Failed to mark orphaned on remove', { path, err });
  }
}

/**
 * При переименовании/перемещении папки пересчитываем категории документов.
 * Записи с category_overridden = true не трогаем.
 */
async function handleFolderRename(oldPath: string, newPath: string): Promise<void> {
  if (!ConnectionManager.exists('kb_main')) return;
  log.info('Folder renamed/moved, recomputing categories', { oldPath, newPath });
  const db = ConnectionManager.get('kb_main');

  try {
    // Найти все документы в старом пути
    const rows = await db.select<Array<{ id: number; file_path: string; metadata: string | null }>>(
      `SELECT id, file_path, metadata FROM library WHERE file_path LIKE $1 ESCAPE '\\'`,
      [`${oldPath.replace(/[%_]/g, (c) => `\\${c}`)}%`],
    );

    for (const row of rows) {
      // Проверяем category_overridden
      let meta: Record<string, unknown> = {};
      try { meta = row.metadata ? JSON.parse(row.metadata) : {}; } catch { /* */ }
      if (meta.category_overridden) continue;

      const newFilePath = newPath + row.file_path.slice(oldPath.length);
      const newCategory = await FolderCategorizerService.categorizeByPath(newFilePath, row.metadata, null);

      await db.execute(
        `UPDATE library SET file_path = $1, folder_path = $2, category = $3 WHERE id = $4`,
        [newFilePath, newFilePath.replace(/\\/g, '/').split('/').slice(0, -1).join('/'), newCategory, row.id],
      );
    }

    if (rows.length > 0) {
      SearchCache.invalidateOnMutation();
      log.info(`Recomputed categories for ${rows.length} documents`);
    }
  } catch (err) {
    log.warn('Failed to recompute categories on rename', err);
  }
}

function scheduleEvent(path: string, kind: 'create' | 'modify' | 'remove' | 'rename'): void {
  const existing = pending.get(path);
  if (existing) clearTimeout(existing.timer);

  const timer = setTimeout(async () => {
    pending.delete(path);
    if (kind === 'create' || kind === 'modify') {
      await handleFileCreate(path);
    } else if (kind === 'remove') {
      await handleFileRemove(path);
    }
    // rename обрабатывается парой событий — см. ниже
  }, DEBOUNCE_MS);

  pending.set(path, { kind, timer });
}

export const FileWatcherService = {
  /**
   * Запускает watcher для корневой папки базы знаний.
   * Безопасно вызывать повторно — повторный вызов останавливает предыдущий.
   */
  async start(kbRootPath: string): Promise<void> {
    if (isRunning) {
      await this.stop();
    }

    try {
      unlisten = await watch(
        kbRootPath,
        (event) => {
          const { type, paths } = event as { type: string; paths: string[] };

          if (!paths || paths.length === 0) return;
          const path = paths[0];

          if (type === 'create') {
            scheduleEvent(path, 'create');
          } else if (type === 'modify') {
            scheduleEvent(path, 'modify');
          } else if (type === 'remove') {
            scheduleEvent(path, 'remove');
          } else if (type === 'rename' && paths.length >= 2) {
            // paths[0] = old, paths[1] = new
            const [oldPath, newPath] = paths;
            // Если это папка (нет расширения) — recompute category
            if (!getExtension(oldPath)) {
              // Debounce с задержкой
              setTimeout(() => handleFolderRename(oldPath, newPath), DEBOUNCE_MS);
            } else {
              // Файл переименован
              scheduleEvent(oldPath, 'remove');
              scheduleEvent(newPath, 'create');
            }
          }
        },
        { recursive: true },
      );
      isRunning = true;
      log.info('FileWatcher started', kbRootPath);
    } catch (err) {
      log.warn('FileWatcher failed to start (watch API may not be available)', err);
      // Не бросаем — watcher опционален
    }
  },

  async stop(): Promise<void> {
    if (unlisten) {
      try { unlisten(); } catch { /* */ }
      unlisten = null;
    }
    // Отмена всех pending debounce
    for (const { timer } of pending.values()) clearTimeout(timer);
    pending.clear();
    isRunning = false;
    log.info('FileWatcher stopped');
  },

  get running(): boolean {
    return isRunning;
  },
};
