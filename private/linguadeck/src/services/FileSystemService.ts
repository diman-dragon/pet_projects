import { join } from "@tauri-apps/api/path";
import { exists, mkdir } from "@tauri-apps/plugin-fs";

const FOLDERS = [
  "database",
  "media",
  "media/audio",
  "media/images",
  "exports",
  "imports",
  "backups"
] as const;

export const FileSystemService = {
  async initLibraryStructure(userSelectedPath: string) {
    const rootPath = await join(userSelectedPath, "Linguadeck");
    if (!(await exists(rootPath))) {
      await mkdir(rootPath, { recursive: true });
    }

    for (const folder of FOLDERS) {
      const target = await join(rootPath, folder);
      if (!(await exists(target))) {
        await mkdir(target, { recursive: true });
      }
    }

    return rootPath;
  },

  async getDatabasePath(libraryPath: string) {
    return join(libraryPath, "database", "main.db");
  },

  async getMediaRoot(libraryPath: string) {
    return join(libraryPath, "media");
  },

  async getAudioFolder(libraryPath: string) {
    return join(libraryPath, "media", "audio");
  },

  async getImagesFolder(libraryPath: string) {
    return join(libraryPath, "media", "images");
  }
};
