import { mkdir, exists, writeFile, readFile, remove, rename, stat } from '@tauri-apps/plugin-fs';
import { join } from '@tauri-apps/api/path';
import { createLogger } from '../logger/Logger';
import { PathAccessService } from './PathAccessService';

const log = createLogger('StorageService');

export const StorageService = {
  async build(basePath: string, schema: string[] | readonly string[] = []) {
    try {
      PathAccessService.assertWorkspacePath(basePath);
      if (!(await exists(basePath))) {
        await mkdir(basePath, { recursive: true });
      }

      for (const folder of schema) {
        const fullPath = await join(basePath, folder);
        PathAccessService.assertWorkspacePath(fullPath);
        if (!(await exists(fullPath))) {
          await mkdir(fullPath, { recursive: true });
        }
      }
    } catch (error) {
      log.error('build failed', { basePath, schema, error });
      throw error;
    }
  },

  async rename(oldPath: string, newPath: string): Promise<void> {
    PathAccessService.assertWorkspaceMove(oldPath, newPath);
    if (!(await exists(oldPath))) {
      throw new Error(`Source path does not exist: ${oldPath}`);
    }
    await rename(oldPath, newPath);
  },

  async saveTextFile(folderPath: string, fileName: string, content: string): Promise<string> {
    const fullPath = await join(folderPath, fileName);
    PathAccessService.assertWorkspacePath(fullPath);
    await writeFile(fullPath, new TextEncoder().encode(content));
    return fullPath;
  },

  async saveFile(folderPath: string, fileName: string, data: ArrayBuffer): Promise<string> {
    PathAccessService.assertWorkspacePath(folderPath);
    if (!(await exists(folderPath))) {
      await mkdir(folderPath, { recursive: true });
    }

    const fullPath = await join(folderPath, fileName);
    PathAccessService.assertWorkspacePath(fullPath);
    await writeFile(fullPath, new Uint8Array(data));
    return fullPath;
  },

  async readTextFile(filePath: string): Promise<string> {
    const content = await this.readFile(filePath);
    return new TextDecoder('utf-8').decode(content);
  },

  async readFile(filePath: string): Promise<Uint8Array> {
    return await readFile(filePath);
  },

  async deleteFile(filePath: string): Promise<void> {
    PathAccessService.assertWorkspacePath(filePath);
    if (await exists(filePath)) {
      await remove(filePath, { recursive: true });
    }
  },

  async secureDelete(filePath: string): Promise<void> {
    PathAccessService.assertWorkspacePath(filePath);
    if (!(await exists(filePath))) return;

    const info = await stat(filePath);
    const size = info.size ?? 0;
    const wipe = new Uint8Array(size);
    await writeFile(filePath, wipe);
    await remove(filePath);
  },

  async pathExists(path: string): Promise<boolean> {
    PathAccessService.assertWorkspacePath(path);
    return await exists(path);
  },

  async getDirName(filePath: string): Promise<string> {
    return await join(filePath, '..');
  },
};
