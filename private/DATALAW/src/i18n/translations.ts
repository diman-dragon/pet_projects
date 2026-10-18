import type { AppLanguage } from './config';
import i18n from './i18n';
import commonRu from './locales/ru/common.json';

export type TranslationKey = keyof typeof commonRu;

export function translate(
  language: AppLanguage,
  key: TranslationKey,
  params?: Record<string, string | number>,
): string {
  const fixed = i18n.getFixedT(language, 'common') as (k: string, opts?: Record<string, string | number>) => string;
  return fixed(key, params);
}
