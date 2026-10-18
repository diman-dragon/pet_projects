// src/pages/knowledge/components/search/PipelineProgress.tsx
// Пошаговый прогресс pipeline поиска

import { CheckCircle2, Loader2, AlertCircle, Circle } from 'lucide-react';
import type { PipelineStep } from '../../../../services/rag/SearchPipeline';

interface PipelineProgressProps {
  steps: PipelineStep[];
}

function StepIcon({ status }: { status: PipelineStep['status'] }) {
  switch (status) {
    case 'running': return <Loader2 size={12} className="animate-spin text-[var(--app-accent)]" />;
    case 'done':    return <CheckCircle2 size={12} className="text-emerald-500" />;
    case 'error':   return <AlertCircle size={12} className="text-red-400" />;
    default:        return <Circle size={12} className="text-black/15" />;
  }
}

export function PipelineProgress({ steps }: PipelineProgressProps) {
  if (!steps.length) return null;

  return (
    <div className="mx-3 mb-2 rounded-xl border border-black/[0.06] bg-white overflow-hidden">
      {steps.map((step, i) => {
        const isLast = i === steps.length - 1;
        return (
          <div
            key={step.id}
            className={`
              flex items-center gap-2.5 px-3 py-2
              ${!isLast ? 'border-b border-black/[0.04]' : ''}
              ${step.status === 'running' ? 'bg-[var(--app-accent)]/[0.02]' : ''}
            `}
          >
            <StepIcon status={step.status} />
            <div className="flex-1 min-w-0">
              <span className={`text-[10px] font-medium ${
                step.status === 'waiting' ? 'text-black/25'
                : step.status === 'running' ? 'text-[var(--app-accent)]'
                : step.status === 'error' ? 'text-red-400'
                : 'text-black/60'
              }`}>
                {step.label}
              </span>
              {step.detail && (
                <span className="ml-1.5 text-[9px] text-black/30 font-mono">
                  {step.detail}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
