// src/pages/knowledge/components/ai/AgentAnswer.tsx
// Блок ответа агента — появляется в правой панели под деталью фрагмента

import { StopCircle, Brain } from 'lucide-react';
import { MarkdownMessage } from '../../../laila/components/shared/MarkdownMessage';
import { StreamingCursor } from '../../../../components/ui/feedback/StreamingCursor';
import type { PipelineStep } from '../../../../services/rag/SearchPipeline';

interface AgentAnswerProps {
  answer: string;
  steps: PipelineStep[];
  isAnalyzing: boolean;
  onAbort?: () => void;
  onLinkClick?: (href: string) => boolean | void;
}

export function AgentAnswer({ answer, steps, isAnalyzing, onAbort, onLinkClick }: AgentAnswerProps) {
  const analyzingStep = steps.find(s => s.id === 'analyzing');
  const hasStarted = analyzingStep && analyzingStep.status !== 'waiting';

  if (!hasStarted) return null;

  return (
    <div className="mt-6 rounded-2xl border border-[var(--app-accent)]/15 bg-gradient-to-b from-[var(--app-accent)]/[0.03] to-white overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2.5 border-b border-[var(--app-accent)]/10 px-4 py-3">
        <div className="relative flex size-6 items-center justify-center">
          {isAnalyzing && (
            <div className="absolute inset-0 rounded-full bg-[var(--app-accent)]/20 animate-ping" style={{ animationDuration: '2s' }} />
          )}
          <div className="relative flex size-6 items-center justify-center rounded-full bg-[var(--app-accent)]/15">
            <Brain size={12} className={`text-[var(--app-accent)] ${isAnalyzing ? 'animate-pulse' : ''}`} />
          </div>
        </div>

        <div>
          <span className="text-[10px] font-black uppercase tracking-widest text-[var(--app-accent)]/80">
            Laila · Анализ документов
          </span>
          <p className="text-[9px] text-black/30">Только по найденным фрагментам</p>
        </div>

        {isAnalyzing && onAbort && (
          <button
            onClick={onAbort}
            className="ml-auto text-black/20 hover:text-black/50 transition-colors"
            title="Остановить"
          >
            <StopCircle size={14} />
          </button>
        )}

        {!isAnalyzing && answer && (
          <span className="ml-auto text-[9px] text-black/20">готово</span>
        )}
      </div>

      {/* Body */}
      <div className="px-4 py-4">
        {answer ? (
          <div className="select-text text-[length:var(--text-xs)] text-black/75 leading-relaxed">
            <MarkdownMessage content={answer} onLinkClick={onLinkClick} />
            {isAnalyzing && <StreamingCursor />}
          </div>
        ) : isAnalyzing ? (
          <div className="flex gap-1 py-2">
            {[0, 150, 300].map(d => (
              <span key={d} className="size-1.5 animate-bounce rounded-full bg-[var(--app-accent)]/40"
                style={{ animationDelay: `${d}ms` }} />
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
