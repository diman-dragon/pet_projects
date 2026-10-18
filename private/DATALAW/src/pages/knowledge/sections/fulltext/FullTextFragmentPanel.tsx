import { BookOpen, ExternalLink, X } from 'lucide-react';
import type { SearchResult } from '../../../../services/rag/UnifiedSearchService';
import type { TranslationKey } from '../../../../i18n/translations';
import { FileExtBadge, MatchBadge, ScoreBar } from './FullTextBadges';
import { buildHighlightedHtml } from './highlight';

interface FullTextFragmentPanelProps {
  result: SearchResult | null;
  selectedIndex: number;
  highlightTerms: string[];
  fragmentRef: React.RefObject<HTMLDivElement | null>;
  onClose: () => void;
  onOpenDocument: (path: string) => void;
  t: (key: TranslationKey, params?: Record<string, string | number>) => string;
}

export function FullTextFragmentPanel({
  result,
  selectedIndex,
  highlightTerms,
  fragmentRef,
  onClose,
  onOpenDocument,
  t,
}: FullTextFragmentPanelProps) {
  const matchLabelMap: Record<'text' | 'vector' | 'hybrid', TranslationKey> = {
    text: 'knowledge.fulltext.matchType.text',
    vector: 'knowledge.fulltext.matchType.vector',
    hybrid: 'knowledge.fulltext.matchType.hybrid',
  };

  return (
    <div
      className={`shrink-0 border-l border-black/[0.06] bg-white transition-all ${result ? 'w-[420px]' : 'w-[240px]'}`}
    >
      <div className="flex items-center justify-between border-b border-black/[0.06] px-4 py-3">
        <div className="flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-[var(--app-accent)]" />
          <span className="text-xs font-semibold text-black/60">
            {result
              ? t('knowledge.fulltext.fragmentPanel.titleWithIndex', { value: selectedIndex + 1 })
              : t('knowledge.fulltext.fragmentPanel.title')}
          </span>
        </div>
        {result ? (
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-black/30 transition-colors hover:bg-black/[0.05] hover:text-black/60"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        ) : null}
      </div>

      <div ref={fragmentRef} className="fts-fragment-panel h-[calc(100%-49px)] overflow-y-auto">
        {result ? (
          <div className="space-y-4 p-4">
            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[var(--app-accent)] text-[11px] font-bold text-white">
                  {selectedIndex + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold leading-tight text-black/80">
                    {result.title}
                  </h3>
                  <p className="mt-1 truncate text-xs text-black/38" title={result.filePath}>
                    {result.filePath}
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-1.5 pl-8">
                <MatchBadge type={result.matchType} label={t(matchLabelMap[result.matchType])} />
                <FileExtBadge filePath={result.filePath} />
                {result.category ? (
                  <span className="rounded bg-black/[0.04] px-1.5 py-0.5 text-[10px] text-black/50">
                    {result.category}
                  </span>
                ) : null}
              </div>
              <div className="pl-8">
                <ScoreBar score={result.score} />
              </div>
            </div>

            <button
              type="button"
              onClick={() => onOpenDocument(result.filePath)}
              className="w-full rounded-2xl border border-amber-200 bg-[#fffdf5] p-4 text-left transition-colors hover:bg-[#fff7db]"
            >
              <p
                className="whitespace-pre-wrap text-sm leading-relaxed text-black/72"
                dangerouslySetInnerHTML={{
                  __html: buildHighlightedHtml(result.chunkText, highlightTerms),
                }}
              />
              <div className="mt-3 flex items-center gap-2 border-t border-amber-200/70 pt-3 text-[11px] font-medium text-amber-700">
                <ExternalLink className="h-3.5 w-3.5" />
                {t('knowledge.fulltext.fragmentPanel.openDocument')}
              </div>
            </button>
          </div>
        ) : (
          <div className="flex h-full flex-col items-center justify-center space-y-3 px-5 text-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-black/[0.03]">
              <BookOpen className="h-4.5 w-4.5 text-black/25" />
            </div>
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-black/35">
                {t('knowledge.fulltext.fragmentPanel.emptyTitle')}
              </p>
              <p className="text-[11px] leading-relaxed text-black/25">
                {t('knowledge.fulltext.fragmentPanel.emptyHint')}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
