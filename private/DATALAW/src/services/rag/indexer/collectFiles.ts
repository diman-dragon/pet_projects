import { join } from '@tauri-apps/api/path';
import { readDir } from '@tauri-apps/plugin-fs';

const SUPPORTED_EXTENSIONS = new Set(['txt', 'md', 'markdown', 'docx', 'pdf']);

export interface IndexedFileEntry {
  fullPath: string;
  name: string;
  category: string;
}

function deriveCategory(baseFolder: string, filePath: string): string {
  const base = baseFolder.replace(/[\\\/]+$/, '').toLowerCase();
  const relative = filePath.toLowerCase().startsWith(base)
    ? filePath.slice(base.length).replace(/^[\\\/]/, '')
    : filePath;
  const parts = relative.split(/[\\\/]/);
  return parts.length > 1 ? parts[0] : 'General';
}

export async function collectFiles(
  folderPath: string,
  signal?: AbortSignal,
): Promise<IndexedFileEntry[]> {
  const files: IndexedFileEntry[] = [];

  const walk = async (dir: string): Promise<void> => {
    const entries = await readDir(dir);
    for (const entry of entries) {
      if (signal?.aborted) return;

      const filePath = await join(dir, entry.name);
      if (entry.isDirectory) {
        await walk(filePath);
        continue;
      }

      const extension = entry.name.split('.').pop()?.toLowerCase() ?? '';
      if (!SUPPORTED_EXTENSIONS.has(extension)) continue;

      files.push({
        fullPath: filePath,
        name: entry.name,
        category: deriveCategory(folderPath, filePath),
      });
    }
  };

  await walk(folderPath);
  return files;
}
