import { ConnectionManager } from '../db/ConnectionManager';
import {
  APP_LANGUAGE_LABELS,
  APP_LANGUAGE_RESPONSE_INSTRUCTIONS,
  isAppLanguage,
  type AppLanguage,
} from '../../i18n/config';
import { createLogger } from '../logger/Logger';

const log = createLogger('AppLanguageService');

let cachedLanguage: AppLanguage | null = null;
let cachedSystemLanguage: AppLanguage | null = null;

function detectSystemLanguage(): AppLanguage {
  if (cachedSystemLanguage) return cachedSystemLanguage;

  const locales =
    typeof navigator !== 'undefined'
      ? [...(navigator.languages ?? []), navigator.language].filter(Boolean)
      : [];

  for (const locale of locales) {
    const normalized = String(locale).toLowerCase();
    if (normalized.startsWith('ru')) {
      cachedSystemLanguage = 'ru';
      return cachedSystemLanguage;
    }
    if (normalized.startsWith('sr')) {
      cachedSystemLanguage = 'sr';
      return cachedSystemLanguage;
    }
    if (normalized.startsWith('en')) {
      cachedSystemLanguage = 'en';
      return cachedSystemLanguage;
    }
  }

  cachedSystemLanguage = 'ru';
  return cachedSystemLanguage;
}

export const LANGUAGE_LABELS = APP_LANGUAGE_LABELS;
export const LANGUAGE_RESPONSE_INSTRUCTIONS = APP_LANGUAGE_RESPONSE_INSTRUCTIONS;

function readI18nLocalStorageLanguage(): AppLanguage | null {
  if (typeof localStorage === 'undefined') return null;
  try {
    const raw = localStorage.getItem('i18nextLng');
    if (isAppLanguage(raw)) return raw;
  } catch {
    /* private mode / denied */
  }
  return null;
}

export const AppLanguageService = {
  async get(): Promise<AppLanguage> {
    if (cachedLanguage) return cachedLanguage;

    const fromLocalStorage = readI18nLocalStorageLanguage();
    if (fromLocalStorage) {
      cachedLanguage = fromLocalStorage;
      return cachedLanguage;
    }

    try {
      const systemLanguage = detectSystemLanguage();
      if (!ConnectionManager.exists('user_system')) {
        cachedLanguage = systemLanguage;
        return cachedLanguage;
      }

      const db = ConnectionManager.get('user_system');
      const rows = await db.select<Array<{ key: string; value: string }>>(
        `SELECT key, value
         FROM app_settings
         WHERE key IN ('ui_language', 'ui_language_source')`,
      );

      const settings = new Map(rows.map((row) => [row.key, row.value]));
      const storedLanguage = settings.get('ui_language');
      const source = settings.get('ui_language_source') === 'manual' ? 'manual' : 'system';

      if (source === 'manual' && isAppLanguage(storedLanguage)) {
        cachedLanguage = storedLanguage;
      } else {
        cachedLanguage = systemLanguage;
        await db.execute(
          `INSERT INTO app_settings (key, value, updated_at)
           VALUES ('ui_language', $1, CURRENT_TIMESTAMP),
                  ('ui_language_source', 'system', CURRENT_TIMESTAMP)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP`,
          [systemLanguage],
        );
      }
    } catch (error) {
      log.warn('Не удалось прочитать язык из БД, используем системный:', error);
      cachedLanguage = detectSystemLanguage();
    }

    return cachedLanguage;
  },

  async set(language: AppLanguage): Promise<void> {
    cachedLanguage = language;

    try {
      if (!ConnectionManager.exists('user_system')) return;

      const db = ConnectionManager.get('user_system');
      await db.execute(
        `INSERT INTO app_settings (key, value, updated_at)
         VALUES ('ui_language', $1, CURRENT_TIMESTAMP)
         ON CONFLICT(key) DO UPDATE SET value = $1, updated_at = CURRENT_TIMESTAMP`,
        [language],
      );
      await db.execute(
        `INSERT INTO app_settings (key, value, updated_at)
         VALUES ('ui_language_source', 'manual', CURRENT_TIMESTAMP)
         ON CONFLICT(key) DO UPDATE SET value = 'manual', updated_at = CURRENT_TIMESTAMP`,
      );
    } catch (error) {
      log.error('Не удалось сохранить язык в БД:', error);
    }
  },

  async resetToSystem(): Promise<AppLanguage> {
    const detected = detectSystemLanguage();
    cachedLanguage = detected;

    try {
      if (!ConnectionManager.exists('user_system')) return detected;

      const db = ConnectionManager.get('user_system');
      await db.execute(
        `INSERT INTO app_settings (key, value, updated_at)
         VALUES ('ui_language', $1, CURRENT_TIMESTAMP),
                ('ui_language_source', 'system', CURRENT_TIMESTAMP)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP`,
        [detected],
      );
    } catch (error) {
      log.error('Не удалось вернуть системный язык:', error);
    }

    return detected;
  },

  getSystemLanguage(): AppLanguage {
    return detectSystemLanguage();
  },

  async getPromptInstruction(): Promise<string> {
    const language = await this.get();
    return APP_LANGUAGE_RESPONSE_INSTRUCTIONS[language];
  },

  getPromptInstructionSync(): string {
    return APP_LANGUAGE_RESPONSE_INSTRUCTIONS[cachedLanguage ?? detectSystemLanguage()];
  },

  clearCache(): void {
    cachedLanguage = null;
    cachedSystemLanguage = null;
  },
};
