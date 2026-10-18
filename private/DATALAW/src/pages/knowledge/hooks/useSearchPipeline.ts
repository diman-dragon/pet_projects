// src/pages/knowledge/hooks/useSearchPipeline.ts
//
// Двухфазная модель поиска.
//
// Фаза 1 (быстрая):
//   ftsPhase = 'done' — FTS-результаты из SearchPipeline уже доступны
//   ftsResults — отображаются немедленно
//
// Фаза 2 (фоновая):
//   phase = 'done' — векторный поиск + AI завершены
//   mergedResults — заменяют ftsResults без сдвига UI
//
// AbortController единый для обеих фаз.
//
// FIX: файл перезаписан с корректной UTF-8 кодировкой.
// Предыдущая версия содержала garbled Windows-1251 символы в комментариях.

import { useCallback, useRef, useState } from 'react';
import {
  runSearchPipeline,
  type PipelineStep,
  type PipelineResult,
} from '../../../services/rag/SearchPipeline';
import type { SearchResult } from '../../../services/rag/UnifiedSearchService';

export type SearchPhase = 'idle' | 'running' | 'done' | 'error';
export type FtsPhase = 'idle' | 'done';

export interface SearchPipelineState {
  phase: SearchPhase;
  /** Статус первой (FTS) фазы */
  ftsPhase: FtsPhase;
  steps: PipelineStep[];
  /** Результаты первой фазы (FTS). Отображаются сразу. */
  ftsResults: SearchResult[];
  vectorResults: SearchResult[];
  textResults: SearchResult[];
  /** Финальные смёрженные результаты (фаза 2). Когда готовы — заменяют ftsResults. */
  mergedResults: SearchResult[];
  agentAnswer: string;
  error: string | null;
}

const INITIAL_STATE: SearchPipelineState = {
  phase: 'idle',
  ftsPhase: 'idle',
  steps: [],
  ftsResults: [],
  vectorResults: [],
  textResults: [],
  mergedResults: [],
  agentAnswer: '',
  error: null,
};

export function useSearchPipeline() {
  const [state, setState] = useState<SearchPipelineState>(INITIAL_STATE);
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const runSearch = useCallback(async (query: string, category?: string) => {
    const trimmed = query.trim();
    if (!trimmed) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setState({ ...INITIAL_STATE, phase: 'running' });
    setSelectedResult(null);

    await runSearchPipeline(
      trimmed,
      category,
      controller.signal,
      {
        onStep: (steps) => {
          setState((prev) => ({ ...prev, steps }));

          // Когда шаг analyzing переходит в done — показываем первую фазу
          const analyzingStep = steps.find((s) => s.id === 'analyzing');
          if (analyzingStep?.status === 'done') {
            setState((prev) => ({ ...prev, ftsPhase: 'done' }));
          }
        },
        onResult: (result: PipelineResult) => {
          setState((prev) => {
            const isFirstResult = prev.ftsResults.length === 0;
            return {
              ...prev,
              // При первом onResult — заполняем ftsResults (FTS-шаг)
              ftsResults: isFirstResult ? result.textResults : prev.ftsResults,
              ftsPhase: isFirstResult ? 'done' : prev.ftsPhase,
              vectorResults: result.vectorResults,
              textResults: result.textResults,
              mergedResults: result.mergedResults,
            };
          });

          setSelectedResult((prev) =>
            prev && result.mergedResults.some((r) => r.chunkId === prev.chunkId)
              ? prev
              : result.mergedResults[0] ?? null,
          );
        },
        onAgentToken: (token) => {
          setState((prev) => ({ ...prev, agentAnswer: prev.agentAnswer + token }));
        },
        onError: (error) => {
          setState((prev) => ({ ...prev, phase: 'error', error }));
        },
      },
      // history не передаём — Search Tab stateless, история только в Semantic Chat
    );

    if (!controller.signal.aborted) {
      setState((prev) => ({ ...prev, phase: 'done' }));
    }
  }, []);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    setState((prev) => ({ ...prev, phase: 'idle' }));
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(INITIAL_STATE);
    setSelectedResult(null);
  }, []);

  /**
   * Результаты для отображения:
   *   - Если вторая фаза готова — mergedResults
   *   - Иначе — ftsResults (быстрые)
   */
  const displayResults =
    state.mergedResults.length > 0 ? state.mergedResults : state.ftsResults;

  return {
    state,
    displayResults,
    selectedResult,
    setSelectedResult,
    runSearch,
    abort,
    reset,
  };
}
