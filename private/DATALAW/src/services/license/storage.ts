// src/services/license/storage.ts
//
// Чтение и запись файла лицензии в защищённую директорию AppConfig.
// Путь определяется Tauri: %APPDATA%\Dorlane-app\license.json (Windows),
// ~/Library/Application Support/Dorlane-app/license.json (macOS),
// ~/.config/Dorlane-app/license.json (Linux).

import { BaseDirectory, exists, readTextFile, remove, writeTextFile } from '@tauri-apps/plugin-fs';
import { LICENSE_FILE, type LicensePayload } from './config';

export async function loadStoredLicense(): Promise<LicensePayload | null> {
  try {
    const hasFile = await exists(LICENSE_FILE, { baseDir: BaseDirectory.AppConfig });
    if (!hasFile) return null;

    const raw = await readTextFile(LICENSE_FILE, { baseDir: BaseDirectory.AppConfig });
    const parsed = JSON.parse(raw) as LicensePayload;

    // Базовая sanity-проверка структуры
    if (typeof parsed !== 'object' || parsed === null || !('key' in parsed)) {
      return null;
    }

    return parsed;
  } catch {
    return null;
  }
}

export async function saveStoredLicense(license: LicensePayload): Promise<void> {
  await writeTextFile(LICENSE_FILE, JSON.stringify(license, null, 2), {
    baseDir: BaseDirectory.AppConfig,
  });
}

export async function removeStoredLicense(): Promise<void> {
  const hasFile = await exists(LICENSE_FILE, { baseDir: BaseDirectory.AppConfig });
  if (!hasFile) return;

  await remove(LICENSE_FILE, { baseDir: BaseDirectory.AppConfig });
}
