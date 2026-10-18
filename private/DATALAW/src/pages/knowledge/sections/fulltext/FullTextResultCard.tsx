import type { SearchResult } from '../../../../services/rag/UnifiedSearchService';
import type { TranslationKey } from '../../../../i18n/translations';
import { FileExtBadge, MatchBadge, ScoreBar } from './FullTextBadges';
import { buildHighlightedHtml } from './highlight';

export function FullTextResultCard({
  result,
  index,
  selected,
  highlightTerms,
  onSelect,
  t,
}: {
  result: SearchResult;
  index: number;
  selected: boolean;
  highlightTerms: string[];
  onSelect: (result: SearchResult) => void;
  t: (key: TranslationKey, params?: Record<string, string | number>) => string;
}) {
  const citationNum = index + 1;
  const normalizedPath = result.filePath.replace(/\\/g, '/');
  const parts = normalizedPath.split('/');
  const fileName = parts[parts.length - 1];
  const dirPath = parts.slice(0, -1).join('/');
  const snippet = result.chunkText.slice(0, 260).replace(/\n+/g, ' ').trim();
  const html = buildHighlightedHtml(`${snippet}${result.chunkText.length > 260 ? '…' : ''}`, highlightTerms);
  const matchLabelMap: Record<SearchResult['matchType'], TranslationKey> = {
    text: 'knowledge.fulltext.matchType.text',
    vector: 'knowledge.fulltext.matchType.vector',
    hybrid: 'knowledge.fulltext.matchType.hybrid',
  };

  return (
    // div вместо button — исключаем вложенные button внутри button
    <div
      role="button"
      tabIndex={0}
      onClick={() => onSelect(result)}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(result); } }}
      className={`group w-full rounded-2xl border p-4 text-left transition-all cursor-pointer ${
        selected
          ? 'border-[var(--app-accent)]/20 bg-[var(--app-accent)]/5 shadow-sm'
          : 'border-transparent hover:border-black/[0.08] hover:bg-white hover:shadow-sm'
      }`}
    >
      <div className="mb-1 flex items-center gap-1 truncate font-mono text-[10px] text-black/35">
        <span className="font-semibold text-[var(--app-accent)]">{fileName}</span>
        {dirPath ? (
          <>
            <span>›</span>
            <span className="truncate">{dirPath}</span>
          </>
        ) : null}
      </div>

      <div className="mb-1.5 flex items-start justify-between gap-3">
        <h3 className="text-[15px] font-medium leading-snug text-[var(--app-accent)] group-hover:underline">
          {result.title}
        </h3>
        <div className="flex shrink-0 items-center gap-1.5">
          <FileExtBadge filePath={result.filePath} />
          <MatchBadge type={result.matchType} label={t(matchLabelMap[result.matchType])} />
        </div>
      </div>

      <div className="text-sm leading-relaxed text-black/60">
        {/* span вместо button — внутри div[role=button] не должно быть button */}
        <span
          role="button"
          tabIndex={0}
          onClick={(e) => { e.stopPropagation(); onSelect(result); }}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.stopPropagation(); onSelect(result); } }}
          className="fts-citation cursor-pointer"
        >
          [{citationNum}]
        </span>{' '}
        <span dangerouslySetInnerHTML={{ __html: html }} />
      </div>

      <div className="mt-3 flex items-center gap-3">
        <ScoreBar score={result.score} />
        {result.category ? (
          <span className="rounded bg-black/[0.04] px-1.5 py-0.5 text-[10px] text-black/45">
            {result.category}
          </span>
        ) : null}
        {result.lang ? (
          <span className="text-[10px] font-mono uppercase text-black/30">{result.lang}</span>
        ) : null}
        <span className="ml-auto text-[10px] text-black/25">
          {t('knowledge.fulltext.fragmentNumber', { value: result.chunkIndex + 1 })}
        </span>
      </div>
    </div>
  );
}
