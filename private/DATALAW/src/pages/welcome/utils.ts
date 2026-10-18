import type { TranslationKey } from '../../i18n/translations';

export function pluralRu(n: number, one: string, few: string, many: string): string {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod100 >= 11 && mod100 <= 19) return many;
  if (mod10 === 1) return one;
  if (mod10 >= 2 && mod10 <= 4) return few;
  return many;
}

export function pluralEmployees(
  language: 'ru' | 'sr' | 'en',
  n: number,
  t: (key: TranslationKey) => string,
): string {
  if (language === 'ru') {
    return pluralRu(
      n,
      t('welcome.inSystemOne'),
      t('welcome.inSystemFew'),
      t('welcome.inSystemMany'),
    );
  }
  return n === 1 ? t('welcome.inSystemOne') : t('welcome.inSystemMany');
}
