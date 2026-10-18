// src/services/license/LicenseService.ts
//
// Главный фасад системы лицензирования.
//
// Ключевые гарантии:
//   1. После успешной activate() лицензия сохраняется на диск — повторный ввод не нужен.
//   2. validate() сначала смотрит в in-memory кеш, затем в файл.
//      При холодном старте файл читается один раз, результат кешируется.
//   3. Если лицензия сохранена и валидна — activate() вызывать не нужно.
//      AppInitializer использует validate() при каждом запуске.

import type { LicensePayload } from './config';
import { getActivationProvider } from './provider';
import { loadStoredLicense, removeStoredLicense, saveStoredLicense } from './storage';

let _cache: LicensePayload | null = null;

export const LicenseService = {
  /**
   * Загружает лицензию с диска (без валидации).
   * Используется только внутри — предпочитайте validate().
   */
  async load(): Promise<LicensePayload | null> {
    if (_cache) return _cache;
    _cache = await loadStoredLicense();
    return _cache;
  },

  /**
   * Сохраняет лицензию на диск и обновляет кеш.
   */
  async save(license: LicensePayload): Promise<void> {
    _cache = license;
    await saveStoredLicense(license);
  },

  /**
   * Активирует лицензию по ключу, сохраняет результат.
   * После этого вызова повторный ввод ключа не требуется.
   */
  async activate(licenseKey: string): Promise<LicensePayload> {
    const license = await getActivationProvider().activate(licenseKey);
    await this.save(license);
    return license;
  },

  /**
   * Валидирует сохранённую лицензию.
   * При каждом запуске AppInitializer вызывает этот метод.
   * Возвращает валидный payload или "пустую" лицензию (valid=false).
   */
  async validate(): Promise<LicensePayload> {
    const stored = await this.load();
    const result = await getActivationProvider().validate(stored);
    // Обновляем кеш результатом валидации (мог измениться valid/activated)
    _cache = result;
    return result;
  },

  /**
   * Сбрасывает кеш (для тестов и при смене воркспейса).
   */
  clearCache(): void {
    _cache = null;
  },

  async reset(): Promise<void> {
    _cache = null;
    await removeStoredLicense();
  },
};
