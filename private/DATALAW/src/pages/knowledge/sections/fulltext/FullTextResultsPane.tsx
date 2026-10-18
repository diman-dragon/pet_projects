import type { SearchResult } from '../../../../services/rag/UnifiedSearchService';
import type { TranslationKey } from '../../../../i18n/translations';
import { FullTextResultCard } from './FullTextResultCard';
import { FullTextResultsEmptyState } from './FullTextResultsEmptyState';

interface FullTextResultsPaneProps {
  results: SearchResult[];
  isSearching: boolean;
  query: string;
  selectedChunkId: number | null;
  highlightTerms: string[];
  onSelect: (result: SearchResult) => void;
  t: (key: TranslationKey, params?: Record<string, string | number>) => string;
}

export function FullTextResultsPane({
  results,
  isSearching,
  query,
  selectedChunkId,
  highlightTerms,
  onSelect,
  t,
}: FullTextResultsPaneProps) {
  if (!results.length) {
    return <FullTextResultsEmptyState query={query} isSearching={isSearching} t={t} />;
  }

  return (
    <div className="max-w-3xl space-y-2 px-6 py-5">
      {results.map((result, index) => (
        <FullTextResultCard
          key={result.chunkId}
          result={result}
          index={index}
          selected={selectedChunkId === result.chunkId}
          highlightTerms={highlightTerms}
          onSelect={onSelect}
          t={t}
        />
      ))}
    </div>
  );
}
