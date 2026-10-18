import type { ExtractedText } from './types';

export function createFallbackText(warning: string): ExtractedText {
  return {
    text: '',
    charCount: 0,
    method: 'fallback',
    warning,
  };
}
