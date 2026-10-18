// src/pages/knowledge/components/results/ChunkDetail.tsx

import { FileText, Hash, Layers } from 'lucide-react';
import type { SearchResult } from '../../../../services/rag/UnifiedSearchService';

import { normalizeTsvText } from '../../../../services/rag/textExtractor/normalizeTsvText';
import { HighlightedText } from './HighlightedText';
// import { AgentAnswer } from '../ai/AgentAnswer';

interface ChunkDetailProps {
  result: SearchResult;
  query: string;
  allResults: SearchResult[];
  vectorResults: SearchResult[];
  // steps: PipelineStep[];
  // agentAnswer: string;
  // isAnalyzing: boolean;
  // onAbort?: () => void;
  // onSelectLink?: (href: string) => boolean | void;
}

export function ChunkDetail({
  result, query, allResults, vectorResults,
  // steps, agentAnswer, isAnalyzing, onAbort, onSelectLink,
}: ChunkDetailProps) {
  const filename = result.filePath.split(/[/\\]/).pop() ?? result.filePath;
  const isVector = vectorResults.some(r => r.chunkId === result.chunkId);
  const scorePct = Math.round(result.score * 100);

  const matchLabel = isVector
    ? (result.matchType === 'hybrid' ? '⚡ Гибридный' : '🔢 Векторный')
    : '🔤 Текстовый';
  const matchColor = isVector ? 'text-violet-500' : 'text-slate-400';

  // Нормализуем сырой TSV/CSV перед отображением
  const displayText = normalizeTsvText(result.chunkText);

  return (
    <div className="mx-auto max-w-2xl">
      {/* Заголовок */}
      <div className="mb-4 flex items-start gap-3">
        <div className={`mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-xl ${isVector ? 'bg-violet-100' : 'bg-slate-100'
          }`}>
          <FileText size={16} className={isVector ? 'text-violet-500' : 'text-slate-400'} />
        </div>
        <div className="flex-1 min-w-0">
          <h2 className="text-[length:var(--text-md)] font-bold text-black/85 leading-snug">{result.title}</h2>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px]">
            <span className="text-black/30 truncate max-w-[200px]" title={result.filePath}>{filename}</span>
            {result.category && (
              <span className="rounded bg-black/[0.05] px-1.5 py-0.5 text-black/40">{result.category}</span>
            )}
          </div>
        </div>
      </div>

      {/* Метрики */}
      <div className="mb-4 flex gap-3">
        <div className="flex-1 rounded-xl border border-black/[0.06] bg-[var(--app-surface-soft)] px-3 py-2.5">
          <p className="text-[9px] uppercase tracking-wider text-black/30 mb-1">Релевантность</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1.5 rounded-full bg-black/[0.06] overflow-hidden">
              <div
                className={`h-full rounded-full ${isVector ? 'bg-violet-400' : 'bg-slate-300'}`}
                style={{ width: `${scorePct}%` }}
              />
            </div>
            <span className={`text-[11px] font-bold ${isVector ? 'text-violet-500' : 'text-slate-400'}`}>
              {scorePct}%
            </span>
          </div>
        </div>
        <div className="rounded-xl border border-black/[0.06] bg-[var(--app-surface-soft)] px-3 py-2.5">
          <p className="text-[9px] uppercase tracking-wider text-black/30 mb-1">Тип</p>
          <span className={`text-[10px] font-semibold ${matchColor}`}>{matchLabel}</span>
        </div>
        <div className="rounded-xl border border-black/[0.06] bg-[var(--app-surface-soft)] px-3 py-2.5">
          <p className="text-[9px] uppercase tracking-wider text-black/30 mb-1">Фрагмент</p>
          <span className="text-[10px] font-semibold text-black/50 flex items-center gap-1">
            <Hash size={9} />{result.chunkIndex + 1}
          </span>
        </div>
      </div>

      {/* Легенда подсветки */}
      {allResults.length > 0 && (
        <div className="mb-3 flex items-center gap-3 text-[9px] text-black/35">
          <Layers size={10} />
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-3 rounded-sm bg-violet-200" /> семантическое совпадение
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-3 rounded-sm bg-amber-200" /> текстовое совпадение
          </span>
        </div>
      )}

      {/* Текст фрагмента (нормализованный) */}
      <div className="rounded-xl border border-black/[0.06] bg-white">
        <div className="flex items-center justify-between border-b border-black/[0.05] px-4 py-2.5">
          <span className="text-[9px] font-semibold uppercase tracking-wider text-black/25">Содержание фрагмента</span>
          <span className="text-[9px] text-black/20">{displayText.length} символов</span>
        </div>
        <div className="p-4 text-[length:var(--text-sm)] text-black/75 leading-relaxed select-text whitespace-pre-wrap">
          <HighlightedText
            text={displayText}
            query={query}
            isVectorResult={isVector}
            matchType={result.matchType}
          />
        </div>
      </div>

      {/* Ответ агента */}
      {/* <AgentAnswer
        answer={agentAnswer}
        steps={steps}
        isAnalyzing={isAnalyzing}
        onAbort={onAbort}
        onLinkClick={onSelectLink}
      /> */}
    </div>
  );
}
