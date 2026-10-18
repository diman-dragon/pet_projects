// src/pages/knowledge/components/results/HighlightedText.tsx
//
// Подсвечивает совпадения с поисковым запросом.
// Разные цвета: жёлтый - текстовое совпадение, фиолетовый - семантическое

import { escapeRegex } from './escapeRegex';

interface HighlightedTextProps {
  text: string;
  query: string;
  isVectorResult?: boolean;
  matchType?: 'vector' | 'text' | 'hybrid';
}

export function HighlightedText({ text, query, isVectorResult = false, matchType = 'text' }: HighlightedTextProps) {
  if (!query.trim()) return <>{text}</>;

  const keywords = query
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 2);

  if (keywords.length === 0) return <>{text}</>;

  const pattern = new RegExp(`(${keywords.map(escapeRegex).join('|')})`, 'i');
  const parts = text.split(pattern);
  const hasMatch = parts.some((part) => pattern.test(part));

  // Выбираем цвет в зависимости от типа совпадения
  const highlightClass =
    matchType === 'vector' || matchType === 'hybrid'
      ? 'rounded px-0.5 text-violet-900 bg-violet-100'
      : 'rounded px-0.5 text-yellow-900 bg-yellow-100';

  const highlightedText = parts.map((part, index) =>
    pattern.test(part) ? (
      <mark key={`${part}-${index}`} className={highlightClass}>
        {part}
      </mark>
    ) : (
      <span key={`${part}-${index}`}>{part}</span>
    ),
  );

  if (isVectorResult && !hasMatch) {
    return (
      <>
        <span className="mb-2 inline-flex items-center gap-1 rounded-md border border-violet-200 bg-violet-50 px-2 py-0.5 text-[9px] font-semibold text-violet-500">
          🔢 Семантическое совпадение — слова запроса не совпадают буквально
        </span>
        <div className="mt-2 rounded-2xl border border-violet-100 bg-violet-50/80 p-3">
          {highlightedText}
        </div>
      </>
    );
  }

  return <>{highlightedText}</>;
}
