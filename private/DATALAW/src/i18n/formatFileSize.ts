import type { TranslationKey } from './translations';

type TFn = (key: TranslationKey, params?: Record<string, string | number>) => string;

export function formatFileSizeWithT(bytes: number, t: TFn): string {
  if (bytes < 1024) return t('format.fileSize.bytes', { n: bytes });
  if (bytes < 1024 * 1024) return t('format.fileSize.kb', { n: (bytes / 1024).toFixed(1) });
  return t('format.fileSize.mb', { n: (bytes / (1024 * 1024)).toFixed(1) });
}
