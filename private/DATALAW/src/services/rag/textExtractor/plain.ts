import { StorageService } from '../../fs/StorageService';
import { normalizeTsvText } from './normalizeTsvText';
import type { ExtractedText } from './types';

export async function extractPlainText(filePath: string): Promise<ExtractedText> {
  const raw = await StorageService.readTextFile(filePath);
  const text = normalizeTsvText(raw);
  return { text, charCount: text.length, method: 'direct' };
}
