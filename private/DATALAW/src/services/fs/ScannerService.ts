// src/services/fs/ScannerService.ts

import { readDir } from '@tauri-apps/plugin-fs';
import { join } from '@tauri-apps/api/path';
import { PathAccessService } from './PathAccessService';

export interface FileNode {
  name: string;
  path: string;
  isFile: boolean;
  children?: FileNode[];
}

export const ScannerService = {
  /**
   * Рекурсивно сканирует директорию и возвращает полное дерево
   */
  async scan(targetPath: string): Promise<FileNode> {
    PathAccessService.assertWorkspacePath(targetPath);
    const entries = await readDir(targetPath);

    const children = await Promise.all(
      entries.map(async (entry) => {
        const fullPath = await join(targetPath, entry.name);

        if (entry.isDirectory) {
          return await this.scan(fullPath);
        }

        return {
          name: entry.name,
          path: fullPath,
          isFile: true,
        };
      }),
    );

    return {
      name: targetPath.split(/[\\/]/).pop() || targetPath,
      path: targetPath,
      isFile: false,
      // Сортируем: сначала папки, потом файлы
      children: children.sort((a, b) => (a.isFile === b.isFile ? 0 : a.isFile ? 1 : -1)),
    };
  },
};
