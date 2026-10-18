import { decodePdfHex, decodePdfLiteral } from './pdfDecode';

function appendArrayText(parts: string[], inner: string): void {
  let text = '';

  const literalPattern = /\(([^)\\]*(?:\\.[^)\\]*)*)\)/g;
  let match: RegExpExecArray | null;
  while ((match = literalPattern.exec(inner)) !== null) {
    text += decodePdfLiteral(match[1]);
  }

  const hexPattern = /<([0-9A-Fa-f]+)>/g;
  while ((match = hexPattern.exec(inner)) !== null) {
    text += decodePdfHex(match[1]);
  }

  if (text.trim()) parts.push(text);
}

export function parsePdfText(raw: string): string {
  const parts: string[] = [];
  const blockPattern = /BT([\s\S]*?)ET/g;
  let blockMatch: RegExpExecArray | null;

  while ((blockMatch = blockPattern.exec(raw)) !== null) {
    const block = blockMatch[1];
    const blockParts: string[] = [];

    const literalPattern = /\(([^)\\]*(?:\\.[^)\\]*)*)\)\s*(?:Tj|TJ)/g;
    let textMatch: RegExpExecArray | null;
    while ((textMatch = literalPattern.exec(block)) !== null) {
      const decoded = decodePdfLiteral(textMatch[1]);
      if (decoded.trim()) blockParts.push(decoded);
    }

    const hexPattern = /<([0-9A-Fa-f]+)>\s*(?:Tj|TJ)/g;
    while ((textMatch = hexPattern.exec(block)) !== null) {
      const decoded = decodePdfHex(textMatch[1]);
      if (decoded.trim()) blockParts.push(decoded);
    }

    const arrayPattern = /\[((?:[^[\]]*(?:\([^)]*\)|<[^>]*>)[^[\]]*)*)\]\s*TJ/g;
    while ((textMatch = arrayPattern.exec(block)) !== null) {
      appendArrayText(blockParts, textMatch[1]);
    }

    if (blockParts.length > 0) parts.push(blockParts.join(' '));
    parts.push('\n');
  }

  return parts
    .join(' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}
