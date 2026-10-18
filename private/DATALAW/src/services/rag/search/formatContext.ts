import type { SearchResult } from '../UnifiedSearchService';

export function formatContext(results: SearchResult[], maxChars = 24000): string {
  if (!results.length) {
    return 'ИНФОРМАЦИЯ В БАЗЕ ЗНАНИЙ ОТСУТСТВУЕТ.';
  }

  let total = 0;
  const blocks = results
    .map((result, index) => {
      const sourceInfo = [
        result.title,
        result.articleRef ? `ст. ${result.articleRef}` : null,
        `chunk ${result.chunkIndex}`
      ].filter(Boolean).join(' | ');

      const block = `--- ФРАГМЕНТ [${index + 1}]: ${sourceInfo} ---\n${result.chunkText}\n\n`;
      if (total + block.length > maxChars) return '';
      total += block.length;
      return block;
    })
    .filter(Boolean);

  return `Используй только следующие фрагменты из базы знаний для формирования ответа:\n\n${blocks.join('')}`;
}
