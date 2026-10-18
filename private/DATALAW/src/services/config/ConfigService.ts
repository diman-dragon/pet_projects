// src/services/config/ConfigService.ts
import { readTextFile, writeTextFile, exists, BaseDirectory, mkdir } from '@tauri-apps/plugin-fs';

const CONFIG_FILE = 'config.json';

export const ConfigService = {
  /**
   * Загружает путь к рабочей директории из конфига
   * @returns string | null
   */
  async load(): Promise<string | null> {
    try {
      const fileExists = await exists(CONFIG_FILE, { baseDir: BaseDirectory.AppConfig });
      if (!fileExists) return null;

      const content = await readTextFile(CONFIG_FILE, { baseDir: BaseDirectory.AppConfig });
      const config = JSON.parse(content);

      return config.workDir || null;
    } catch (err) {
      console.error('Config Load Error:', err);
      return null;
    }
  },

  /**
   * Сохраняет путь к Dorlane в конфиг
   */
  async save(workDir: string): Promise<boolean> {
    try {
      // Убеждаемся, что папка для конфига существует (AppData/Roaming/имя_апп)
      const dirExists = await exists('', { baseDir: BaseDirectory.AppConfig });
      if (!dirExists) {
        await mkdir('', { baseDir: BaseDirectory.AppConfig, recursive: true });
      }

      const config = { workDir, lastSync: new Date().toISOString() };
      await writeTextFile(CONFIG_FILE, JSON.stringify(config, null, 2), {
        baseDir: BaseDirectory.AppConfig,
      });

      return true;
    } catch (err) {
      console.error('Config Save Error:', err);
      return false;
    }
  },

  /**
   * Вспомогательный метод (если нужен просто факт наличия)
   */
  async checkExists(): Promise<boolean> {
    const path = await this.load();
    return path !== null;
  },
};
