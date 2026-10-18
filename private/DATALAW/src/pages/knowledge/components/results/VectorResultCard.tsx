import type { SearchResult } from '../../../../services/rag/UnifiedSearchService';
import { normalizeTsvText } from '../../../../services/rag/textExtractor/normalizeTsvText';
import { HighlightedText } from './HighlightedText';

interface VectorResultCardProps {
  result: SearchResult;
  index: number;
  query: string;
  isSelected: boolean;
  onSelect: (result: SearchResult) => void;
  isVectorMatch: boolean;
}

function MatchPill({ type, isVector }: { type: SearchResult['matchType']; isVector: boolean }) {
  if (type === 'hybrid' || (type === 'vector' && isVector)) return (
    <span className="rounded px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider bg-violet-100 text-violet-500">
      ⚡ семантика
    </span>
  );
  return (
    <span className="rounded px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider bg-slate-100 text-slate-400">
      🔤 текст
    </span>
  );
}

function ScoreBar({ score, isVector }: { score: number; isVector: boolean }) {
  const pct = Math.round(score * 100);
  const barColor = isVector
    ? (pct >= 60 ? 'bg-violet-400' : pct >= 40 ? 'bg-violet-300' : 'bg-violet-200')
    : 'bg-slate-300';
  return (
    <div className="flex items-center gap-1.5">
      <div className="h-1 w-10 rounded-full bg-black/[0.06] overflow-hidden">
        <div className={`h-full rounded-full ${barColor}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-[9px] font-mono ${isVector ? 'text-violet-400' : 'text-black/30'}`}>
        {pct}%
      </span>
    </div>
  );
}

export function VectorResultCard({ result, index, query, isSelected, onSelect, isVectorMatch }: VectorResultCardProps) {
  const rawText = normalizeTsvText(result.chunkText);
  const snippet = rawText.length > 180
    ? rawText.slice(0, 180).trimEnd() + '…'
    : rawText;
  const filename = result.filePath.split(/[/\\]/).pop() ?? '';

  return (
    <div
      onClick={() => onSelect(result)}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && onSelect(result)}
      className={`
        group mb-2 cursor-pointer rounded-xl border p-3 transition-all duration-150
        ${isSelected
          ? isVectorMatch
            ? 'border-violet-200 bg-violet-50/50 shadow-sm'
            : 'border-[var(--app-accent)]/20 bg-[var(--app-accent)]/[0.03] shadow-sm'
          : 'border-black/[0.06] bg-white hover:border-black/[0.12]'
        }
      `}
    >
      <div className="flex items-start gap-2 mb-1.5">
        <span className={`
          shrink-0 mt-0.5 flex size-5 items-center justify-center rounded-md text-[9px] font-black
          ${isSelected
            ? isVectorMatch ? 'bg-violet-500 text-white' : 'bg-[var(--app-accent)] text-white'
            : 'bg-black/[0.05] text-black/35'
          }
        `}>
          {index + 1}
        </span>
        <span className="text-[length:var(--text-xs)] font-semibold text-black/80 line-clamp-1 leading-snug">
          {result.title}
        </span>
      </div>

      <div className="text-[10px] text-black/45 leading-relaxed line-clamp-2 mb-2 pl-7">
        <HighlightedText text={snippet} query={query} isVectorResult={isVectorMatch} matchType={result.matchType} />
      </div>

      <div className="flex items-center gap-2 pl-7">
        <MatchPill type={result.matchType} isVector={isVectorMatch} />
        <ScoreBar score={result.score} isVector={isVectorMatch} />
        <span className="ml-auto text-[9px] text-black/20 truncate max-w-[90px]" title={result.filePath}>
          {filename}
        </span>
      </div>
    </div>
  );
}
