import { basename, extname, join } from '@tauri-apps/api/path';
import { exists, readDir } from '@tauri-apps/plugin-fs';
import { AppPaths } from '../../config/AppPaths';
import { ConnectionManager } from '../../db/ConnectionManager';
import { createLogger } from '../../logger/Logger';
import { SUPPORTED_EXTENSIONS } from './constants';
import { getAllLibraryDocuments } from './documentReads';

const log = createLogger('KnowledgeLibrary');

function normalizeExtension(filePath: string) {
  return extname(filePath).then((value) => {
    const normalized = value.toLowerCase();
    return normalized.startsWith('.') ? normalized : `.${normalized}`;
  });
}

export async function scanAndAddKnowledgeFolder(): Promise<string[]> {
  const kbPath = AppPaths.getKbPath();
  if (!kbPath) {
    log.error('[SCAN] KB path not initialized');
    return [];
  }

  const dataRowPath = await join(kbPath, 'DATA_ROW');
  if (!(await exists(dataRowPath))) {
    log.info('[SCAN] DATA_ROW folder does not exist, creating it');
    try {
      await import('@tauri-apps/plugin-fs').then((fs) => fs.mkdir(dataRowPath));
    } catch (e) {
      log.error('[SCAN] Failed to create DATA_ROW folder', e);
      return [];
    }
  }

  const existingDocs = await getAllLibraryDocuments();
  const existingPaths = new Set(existingDocs.map((doc) => doc.file_path.toLowerCase()));
  const entries = await readDir(dataRowPath);
  log.info(`[SCAN] Found ${entries.length} entries`);
  const db = ConnectionManager.get('kb_main');
  const addedFiles: string[] = [];

  for (const entry of entries) {
    log.debug(`[SCAN] Found file: ${entry.name}`);
    if (entry.isDirectory) {
      continue;
    }

    const filePath = await join(dataRowPath, entry.name);
    const extension = await normalizeExtension(filePath);
    if (!SUPPORTED_EXTENSIONS.includes(extension)) {
      continue;
    }

    if (existingPaths.has(filePath.toLowerCase())) {
      continue;
    }

    try {
      const title = await basename(filePath);
      await db.execute(
        `INSERT OR IGNORE INTO library (title, file_path, file_type, category, hash, metadata, created_at, updated_at)
         VALUES ($1, $2, $3, $4, NULL, NULL, datetime('now'), datetime('now'))`,
        [title, filePath, extension, 'Входящие'],
      );

      const [{ count }] = await db.select<{ count: number }[]>(
        `SELECT COUNT(*) as count FROM library WHERE file_path = $1`,
        [filePath],
      );
      if (count > 0) {
        addedFiles.push(entry.name);
      }
    } catch (error) {
      log.error(`[scanAndAddFolder] Error processing file ${filePath}:`, error);
    }
  }

  return addedFiles;
}
