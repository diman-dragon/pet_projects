import { invoke } from '@tauri-apps/api/core';
import { createFallbackText } from './fallback';
import type { ExtractedText } from './types';

export async function extractDocx(filePath: string): Promise<ExtractedText> {
  try {
    // Use Tauri backend for proper DOCX extraction
    const text = await invoke<string>('extract_document_text', { filePath });
    return { text, charCount: text.length, method: 'direct' };
  } catch (error: any) {
    return createFallbackText(`Не удалось извлечь текст из DOCX: ${error?.message ?? error}`);
  }
}
