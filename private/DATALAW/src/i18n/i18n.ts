import i18n from 'i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import { initReactI18next } from 'react-i18next';
import commonEn from './locales/en/common.json';
import commonRu from './locales/ru/common.json';
import commonSr from './locales/sr/common.json';
import { isAppLanguage, type AppLanguage } from './config';

/** Standard i18next localStorage key — LanguageDetector reads/writes this. */
export const I18N_STORAGE_KEY = 'i18nextLng';

export function normalizeAppLanguage(lng: string | undefined): AppLanguage {
  if (!lng) return 'ru';
  const code = lng.split('-')[0]?.toLowerCase() ?? 'ru';
  return isAppLanguage(code) ? code : 'ru';
}

void i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      ru: { common: commonRu },
      en: { common: commonEn },
      sr: { common: commonSr },
    },
    fallbackLng: 'ru',
    supportedLngs: ['ru', 'en', 'sr'],
    ns: ['common'],
    defaultNS: 'common',
    interpolation: {
      escapeValue: false,
      prefix: '{',
      suffix: '}',
    },
    detection: {
      order: ['localStorage', 'navigator'],
      caches: ['localStorage'],
      lookupLocalStorage: I18N_STORAGE_KEY,
    },
    load: 'languageOnly',
    react: { useSuspense: false },
  });

i18n.on('languageChanged', (lng) => {
  const next = normalizeAppLanguage(lng);
  if (lng !== next) {
    void i18n.changeLanguage(next);
  }
});

export default i18n;
