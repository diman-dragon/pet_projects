import type { SearchResult } from '../../../../services/rag/UnifiedSearchService';
import { WelcomeState } from '../WelcomeState';
import { EmptyResultsState } from './EmptyResultsState';
import { VectorResultCard } from './VectorResultCard';

interface ResultsListProps {
  phase: 'idle' | 'running' | 'done' | 'error';
  results: SearchResult[];
  vectorResults: SearchResult[];
  selectedResult: SearchResult | null;
  onSelect: (r: SearchResult) => void;
  hasEmbeddings: boolean;
  query: string;
}

export function ResultsList({
  phase, results, vectorResults,
  selectedResult, onSelect, hasEmbeddings, query,
}: ResultsListProps) {
  if (phase === 'idle') return <WelcomeState />;
  if (phase === 'running' && results.length === 0) return null;
  if (phase === 'done' && results.length === 0) return <EmptyResultsState hasIndex={hasEmbeddings} />;

  const vectorIds = new Set(vectorResults.map(r => r.chunkId));

  return (
    <div>
      {results.map((result, index) => (
        <VectorResultCard
          key={result.chunkId}
          result={result}
          index={index}
          query={query}
          isSelected={selectedResult?.chunkId === result.chunkId}
          onSelect={onSelect}
          isVectorMatch={vectorIds.has(result.chunkId)}
        />
      ))}
    </div>
  );
}
