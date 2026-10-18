import { appConfigDir, join } from "@tauri-apps/api/path";
import { exists, mkdir, readTextFile, writeTextFile } from "@tauri-apps/plugin-fs";

const CONFIG_NAME = "config.json";

interface AppConfigFile {
  libraryPath: string | null;
}

async function getConfigFilePath() {
  const configDir = await appConfigDir();
  if (!(await exists(configDir))) {
    await mkdir(configDir, { recursive: true });
  }
  return join(configDir, CONFIG_NAME);
}

export const ConfigService = {
  async saveLibraryPath(path: string) {
    const filePath = await getConfigFilePath();
    const payload: AppConfigFile = { libraryPath: path };
    await writeTextFile(filePath, JSON.stringify(payload, null, 2));
  },

  async getSavedLibraryPath(): Promise<string | null> {
    try {
      const filePath = await getConfigFilePath();
      if (!(await exists(filePath))) {
        return null;
      }

      const raw = await readTextFile(filePath);
      const parsed = JSON.parse(raw) as AppConfigFile;
      return parsed.libraryPath ?? null;
    } catch {
      return null;
    }
  },

  async clearLibraryPath() {
    const filePath = await getConfigFilePath();
    const payload: AppConfigFile = { libraryPath: null };
    await writeTextFile(filePath, JSON.stringify(payload, null, 2));
  }
};
