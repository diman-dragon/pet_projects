// src/pages/laila/components/analyze/AnalyzePanel.tsx

import { AnalysisResult } from '../../types';
import { AnalyzeTextInput } from './AnalyzeTextInput';
import { AnalyzeEmpty } from './AnalyzeEmpty';
import { AnalyzeLoader } from './AnalyzeLoader';
import { AnalyzeResult } from './AnalyzeResult';
import { RotateCcw } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { StreamingCursor } from '../../../../components/ui/feedback/StreamingCursor';

interface AnalyzePanelProps {
  text: string;
  onTextChange: (v: string) => void;
  onRun: () => void;
  onClear: () => void;
  isAnalyzing: boolean;
  result: AnalysisResult | null;
  streamingText?: string;
}

export function AnalyzePanel({
  text,
  onTextChange,
  onRun,
  onClear,
  isAnalyzing,
  result,
  streamingText,
}: AnalyzePanelProps) {
  const { t } = useI18n();
  const hasContent = result || (isAnalyzing && streamingText);

  return (
    <div className="flex h-full overflow-hidden">
      <AnalyzeTextInput
        value={text}
        onChange={onTextChange}
        onRun={onRun}
        isAnalyzing={isAnalyzing}
      />

      <div className="w-1/2 flex flex-col overflow-hidden bg-white border-l border-black/[0.05]">
        <div className="flex items-center justify-between px-5 py-3 border-b border-black/[0.05]">
          <span className="text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/30">
            {t('analyze.title')}
          </span>
          {hasContent && (
            <button
              onClick={onClear}
              className="flex items-center gap-1.5 px-2 py-1 rounded hover:bg-black/[0.05] text-black/40 hover:text-black transition-all group"
            >
              <span className="text-[length:var(--text-xs)] font-bold uppercase opacity-0 group-hover:opacity-100 transition-opacity">
                {t('analyze.clear')}
              </span>
              <RotateCcw size={14} />
            </button>
          )}
        </div>

        <div className="flex-1 p-5 gap-4 overflow-y-auto custom-scroll">
          {/* 1. Состояние покоя */}
          {!hasContent && !isAnalyzing && <AnalyzeEmpty />}

          {/* 2. Загрузка (самый старт, когда токенов еще нет) */}
          {isAnalyzing && !streamingText && <AnalyzeLoader />}

          {/* 3. Красивый вывод (результат ИЛИ активный стриминг) */}
          {isAnalyzing && !result && streamingText && (
            <div className="max-h-full overflow-auto rounded-xl border border-black/[0.06] bg-black/[0.02] p-4 font-[var(--font-mono)] text-[length:var(--text-xs)] leading-relaxed text-black/70 opacity-70">
              <pre className="whitespace-pre-wrap break-words">{streamingText}<StreamingCursor /></pre>
            </div>
          )}

          {hasContent && !(isAnalyzing && !result && streamingText) && (
            <AnalyzeResult
              result={result}
              isStreaming={isAnalyzing}
              rawStreamingText={streamingText}
            />
          )}
        </div>
      </div>
    </div>
  );
}
