// src/services/utils/NamingService.ts
// Общие утилиты именования. Специфичные функции вынесены в отдельные сервисы.

import { DateFormatService } from './DateFormatService';
import { TranslitService } from './TranslitService';

export const NamingService = {
  /** Блок YYMM_NN (например, 2604_01) */
  generateIdBlock: DateFormatService.toIdBlock,

  /** Дата YYMMDD */
  getShortDate: DateFormatService.toShortDate,

  /** Сборщик TYPE_ID_NAME_STATUS в UPPERCASE */
  formatFolderName(type: string, idBlock: string, name: string, status: string): string {
    return `${type.toUpperCase()}_${idBlock}_${name.toUpperCase()}_${status.toUpperCase()}`;
  },

  /** Юрлица: "АО Вектор" → "AO_VEKTOR" */
  toLegalName(text: string): string {
    const str = String(text || '').trim();
    if (!str) return '';

    const result = TranslitService.transliterate(str)
      .toUpperCase()
      .replace(/\s+/g, '_')
      .replace(/[^A-Z0-9_]/g, '')
      .replace(/_+/g, '_')
      .replace(/(^_|_$)/g, '');

    return result.substring(0, 40);
  },

  /** Физлица: "Иванов Иван" → "IVANOV_I_I" */
  toShortName(text: string): string {
    const parts = String(text || '')
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    if (!parts.length) return '';

    const lastName = TranslitService.transliterate(parts[0]).toUpperCase();
    const initials = parts
      .slice(1)
      .map((p) => {
        const firstChar = p.charAt(0);
        return TranslitService.transliterate(firstChar).toUpperCase();
      })
      .filter(Boolean);

    const result = initials.length ? `${lastName}_${initials.join('_')}` : lastName;
    return result.substring(0, 30);
  },

  /** Стандартное имя (дело, линия и т.д.): "Дело №1" → "DELO_1" */
  toStandardName(text: string): string {
    const str = String(text || '').trim();
    if (!str) return '';

    const result = TranslitService.transliterate(str)
      .toUpperCase()
      .replace(/\s+/g, '_')
      .replace(/[^A-Z0-9_]/g, '')
      .replace(/_+/g, '_')
      .replace(/(^_|_$)/g, '');

    return result.substring(0, 40);
  },
};
