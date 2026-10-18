import { extractDocx } from './textExtractor/docx';
import { createFallbackText } from './textExtractor/fallback';
import { extractPdf } from './textExtractor/pdf';
import { extractPlainText } from './textExtractor/plain';
import type { ExtractedText } from './textExtractor/types';

async function extractByExtension(filePath: string, extension: string): Promise<ExtractedText> {
  switch (extension) {
    case 'txt':
    case 'md':
    case 'markdown':
      return extractPlainText(filePath);
    case 'docx':
      return extractDocx(filePath);
    case 'pdf':
      return extractPdf(filePath);
    default:
      return extractPlainText(filePath).catch(() =>
        createFallbackText(`Неподдерживаемый формат: .${extension}`),
      );
  }
}

export const TextExtractorService = {
  async extract(filePath: string): Promise<ExtractedText> {
    const extension = filePath.split('.').pop()?.toLowerCase() ?? '';
    return extractByExtension(filePath, extension);
  },
};

export type { ExtractedText };
