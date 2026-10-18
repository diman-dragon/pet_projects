function decodeXmlEntities(text: string): string {
  return text
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCodePoint(Number.parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec) => String.fromCodePoint(Number.parseInt(dec, 10)))
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&apos;/g, "'")
    .replace(/&quot;/g, '"');
}

/**
 * Проверяет, имеет ли следующий <w:t> после позиции атрибут xml:space="preserve".
 * Если да — не добавляем пробел (текст уже содержит нужные пробелы).
 */
function nextWtHasPreserve(xml: string, pos: number): boolean {
  const nextWtMatch = xml.slice(pos).match(/<w:t\b([^>]*)>/);
  if (!nextWtMatch) return false;
  return /xml:space\s*=\s*["']preserve["']/.test(nextWtMatch[1]);
}

export function docxXmlToText(xmlText: string): string {
  let decoded = decodeXmlEntities(xmlText);

  // Заменяем табы, переносы строк
  decoded = decoded
    .replace(/<w:tab\b[^>]*\/?>/g, '\t')
    .replace(/<w:br\b[^>]*\/?>/g, '\n')
    .replace(/<w:cr\b[^>]*\/?>/g, '\n')
    .replace(/<\/w:p>/g, '\n\n')
    .replace(/<w:p\b[^>]*>/g, '');

  // Обрабатываем w:r элементы: добавляем пробел между </w:r> и следующим <w:r>
  // только если следующий <w:t> НЕ имеет xml:space="preserve"
  let result = '';
  let i = 0;
  while (i < decoded.length) {
    const endRunIndex = decoded.indexOf('</w:r>', i);
    if (endRunIndex === -1) {
      result += decoded.slice(i);
      break;
    }

    // Добавляем всё до </w:r>
    result += decoded.slice(i, endRunIndex + 6); // +6 for '</w:r>'
    i = endRunIndex + 6;

    // Проверяем, есть ли следующий <w:r>
    const nextRunMatch = decoded.slice(i).match(/^\s*<w:r\b/);
    if (nextRunMatch) {
      // Смотрим, имеет ли следующий <w:t> xml:space="preserve"
      if (!nextWtHasPreserve(decoded, i)) {
        result += ' '; // Добавляем пробел только если нет preserve
      }
    }
  }

  // Убираем оставшиеся XML теги
  return result
    .replace(/<w:t\b[^>]*>/g, '')
    .replace(/<\/w:t>/g, '')
    .replace(/<[^>]+>/g, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n[ \t]+/g, '\n')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}
