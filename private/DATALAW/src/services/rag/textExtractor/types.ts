export interface ExtractedText {
  text: string;
  charCount: number;
  method: 'direct' | 'docx-xml' | 'pdf-text' | 'fallback';
  warning?: string;
}
