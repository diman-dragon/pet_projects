// src/services/utils/TranslitService.ts
// Ответственность: кириллица → латиница с сохранением регистра.
// Никакой логики именования папок — только символьное преобразование.

const CYRILLIC_MAP: Record<string, string> = {
  а: 'a',
  б: 'b',
  в: 'v',
  г: 'g',
  д: 'd',
  е: 'e',
  ё: 'yo',
  ж: 'zh',
  з: 'z',
  и: 'i',
  й: 'i',
  к: 'k',
  л: 'l',
  м: 'm',
  н: 'n',
  о: 'o',
  п: 'p',
  р: 'r',
  с: 's',
  т: 't',
  у: 'u',
  ф: 'f',
  х: 'kh',
  ц: 'ts',
  ч: 'ch',
  ш: 'sh',
  щ: 'shch',
  ъ: '',
  ы: 'y',
  ь: '',
  э: 'e',
  ю: 'yu',
  я: 'ya',
};

export const TranslitService = {
  /**
   * Транслитерирует строку: кириллица → латиница.
   * Сохраняет регистр символов для корректной работы NamingService.
   *
   * @example transliterate("Вектор") → "Vektor"
   * @example transliterate("ООО") → "OOO"
   */
  transliterate(text: string): string {
    if (!text?.trim()) return 'unknown';

    return text
      .split('')
      .map((char) => {
        const lowChar = char.toLowerCase();
        const isUpper = char !== lowChar;
        const mapRes = CYRILLIC_MAP[lowChar];

        if (mapRes !== undefined) {
          // Возвращаем транслит в нужном регистре
          return isUpper ? mapRes.toUpperCase() : mapRes;
        }

        // Исправлено: разрешаем A-Z, чтобы заглавная латиница не превращалась в "_"
        return /[a-zA-Z0-9]/.test(char) ? char : '_';
      })
      .join('')
      .replace(/_+/g, '_')
      .replace(/^_|_$/g, '');
  },
};
