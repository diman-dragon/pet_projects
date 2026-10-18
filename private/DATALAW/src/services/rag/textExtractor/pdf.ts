import { invoke } from '@tauri-apps/api/core';
import type { ExtractedText } from './types';

export async function extractPdf(filePath: string): Promise<ExtractedText> {
  try {
    // Use Tauri backend for proper PDF extraction
    const text = await invoke<string>('extract_document_text', { filePath });

    if (!text || text.length < 50) {
      return {
        text: text || '',
        charCount: text?.length ?? 0,
        method: 'pdf-text',
        warning:
          text && text.length > 0
            ? 'PDF может быть сканированным: текст извлечен не полностью'
            : 'Не удалось извлечь текст из PDF',
      };
    }

    return { text, charCount: text.length, method: 'pdf-text' };
  } catch (error: any) {
    return {
      text: '',
      charCount: 0,
      method: 'fallback',
      warning: `Не удалось прочитать PDF: ${error?.message ?? error}`,
    };
  }
}
