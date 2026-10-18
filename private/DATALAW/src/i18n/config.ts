export const SUPPORTED_APP_LANGUAGES = ['ru', 'en', 'sr'] as const;

export type AppLanguage = (typeof SUPPORTED_APP_LANGUAGES)[number];

export const APP_LANGUAGE_LABELS: Record<AppLanguage, string> = {
  ru: 'Русский',
  sr: 'Srpski (latinica)',
  en: 'English',
};

export const APP_LANGUAGE_RESPONSE_INSTRUCTIONS: Record<AppLanguage, string> = {
  ru: `ЯЗЫК ОТВЕТА: По умолчанию отвечай на русском языке, то есть на языке интерфейса и системы.
Если пользователь явно просит другой язык, обязательно ответь на запрошенном языке.
Не переключай язык из-за языка документов или цитат.
Иностранные термины, названия дел и цитаты из иностранных источников давай в оригинале с переводом в скобках.`,

  sr: `JEZIK ODGOVORA: Podrazumevano odgovaraj na srpskom jeziku (latinica), odnosno na jeziku interfejsa i sistema.
Ako korisnik izričito traži drugi jezik, obavezno odgovori na traženom jeziku.
Ne menjaj jezik samo zato što su dokumenti ili citati na drugom jeziku.
Strane termine, nazive predmeta i citate navodi u originalu sa prevodom u zagradama.`,

  en: `RESPONSE LANGUAGE: By default, answer in English, which is the interface and system language.
If the user explicitly asks for another language, answer in that requested language.
Do not switch languages because source documents or quotations use a different language.
Foreign-language terms, case names, and citations should be given in the original with translation in parentheses.`,
};

export function isAppLanguage(value: string | undefined | null): value is AppLanguage {
  return SUPPORTED_APP_LANGUAGES.includes(value as AppLanguage);
}

export function appLanguageToBcp47(lang: AppLanguage): string {
  switch (lang) {
    case 'ru':
      return 'ru-RU';
    case 'sr':
      return 'sr-Latn-RS';
    case 'en':
    default:
      return 'en-US';
  }
}
