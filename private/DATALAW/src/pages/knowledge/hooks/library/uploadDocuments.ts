import { open as dialogOpen } from '@tauri-apps/plugin-dialog';
import { join } from '@tauri-apps/api/path';
import { readFile } from '@tauri-apps/plugin-fs';
import { AppPaths } from '../../../../services/config/AppPaths';
import { StorageService } from '../../../../services/fs/StorageService';
import { IndexerService } from '../../../../services/rag/IndexerService';

export async function uploadKnowledgeDocuments() {
  const selected = await dialogOpen({
    multiple: true,
    filters: [
      { name: 'Documents', extensions: ['pdf', 'doc', 'docx', 'txt', 'rtf', 'html', 'htm'] },
      { name: 'All files', extensions: ['*'] },
    ],
  });

  if (!selected || (Array.isArray(selected) && selected.length === 0)) {
    return false;
  }

  const files = Array.isArray(selected) ? selected : [selected];
  const kbPath = AppPaths.getKbPath();
  if (!kbPath) {
    throw new Error('Knowledge_Base path not set. Please initialize workspace first.');
  }

  const dataRowPath = await join(kbPath, 'DATA_ROW');
  await StorageService.build(dataRowPath, []);

  for (const filePath of files) {
    try {
      const fileName = filePath.split(/[/\\]/).pop() || 'unnamed';
      const fileData = await readFile(filePath);
      const targetPath = await StorageService.saveFile(dataRowPath, fileName, fileData.buffer);

      await IndexerService.indexFile({
        filePath: targetPath,
        title: fileName.replace(/\.[^/.]+$/, ''),
        category: 'Входящие', // Обеспечить наличие этой папки в настройках
      });
    } catch (error) {
      console.error(`[LIBRARY] Failed to process file: ${filePath}`, error);
    }
  }

  return true;
}
