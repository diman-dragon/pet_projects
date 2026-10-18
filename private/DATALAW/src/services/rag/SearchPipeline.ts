// src/services/rag/SearchPipeline.ts
//
// ИТЕРАЦИЯ 6: Убран дублирующий вызов processQuery.
// understandQuery вызывается один раз — внутри KnowledgeSearchAgent.search().
// SearchPipeline больше не вызывает processQuery отдельно, queryLang
// приходит через колбэк onQueryLang от агента.
//
// Изменения:
//   - Убран processQuery (дублировал understandQuery → двойной LLM-вызов)
//   - Добавлен колбэк onQueryLang в PipelineCallbacks
//   - history передаётся в KnowledgeSearchAgent.search()

import { createLogger } from '../logger/Logger';
import type { SearchResult } from './UnifiedSearchService';

const log = createLogger('SearchPipeline');

export type PipelineStepId = 'idle' | 'analyzing' | 'done' | 'error';

export interface PipelineStep {
  id: PipelineStepId;
  label: string;
  status: 'waiting' | 'running' | 'done' | 'error';
  detail?: string;
}

export interface PipelineResult {
  query: string;
  queryVector: Float32Array | null;
  vectorResults: SearchResult[];
  textResults: SearchResult[];
  mergedResults: SearchResult[];
  agentAnswer: string;
  queryId: number | null;
  queryLang?: string;
}

export interface PipelineCallbacks {
  onStep: (steps: PipelineStep[]) => void;
  onResult: (result: PipelineResult) => void;
  onError: (error: string) => void;
  onAgentToken: (token: string) => void;
  // FIX: колбэк для передачи определённого языка запроса вызывающему компоненту
  onQueryLang?: (lang: string) => void;
}

// FIX: history передаётся снаружи (из UI-компонента KnowledgeHub)
// для поддержки многоходовых сессий без потери контекста.
export interface HistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

const STEPS_INIT: PipelineStep[] = [
  { id: 'analyzing', label: 'Анализ документов (Laila)', status: 'waiting' },
];

function cloneSteps(steps: PipelineStep[]): PipelineStep[] {
  return steps.map((s) => ({ ...s }));
}

function setStep(
  steps: PipelineStep[],
  id: PipelineStepId,
  status: PipelineStep['status'],
  detail?: string,
): PipelineStep[] {
  const next = cloneSteps(steps);
  const s = next.find((x) => x.id === id);
  if (s) {
    s.status = status;
    if (detail !== undefined) s.detail = detail;
  }
  return next;
}

export async function runSearchPipeline(
  query: string,
  _category: string | undefined,
  signal: AbortSignal,
  callbacks: PipelineCallbacks,
  history: HistoryMessage[] = [],
): Promise<void> {
  let steps = cloneSteps(STEPS_INIT);
  const emit = () => callbacks.onStep(cloneSteps(steps));

  const result: PipelineResult = {
    query,
    queryVector: null,
    vectorResults: [],
    textResults: [],
    mergedResults: [],
    agentAnswer: '',
    queryId: null,
    queryLang: undefined,
  };

  try {
    // ── Запуск агента Laila ────────────────────────────────────────────────
    // FIX: queryLang НЕ определяется здесь отдельно — это делает KnowledgeSearchAgent
    // внутри через understandQuery. Убираем дублирующий processQuery.
    steps = setStep(steps, 'analyzing', 'running');
    emit();

    try {
      const { KnowledgeSearchAgent } = await import('../ai/agents/KnowledgeSearchAgent');
      const searchResult = await KnowledgeSearchAgent.search(
        query,
        (token) => {
          if (signal.aborted) return;
          result.agentAnswer += token;
          callbacks.onAgentToken(token);
        },
        signal,
        undefined,     // chatId
        undefined,     // queryLang — агент определяет сам через understandQuery
        history,       // FIX: передаём историю для многоходовых сессий
      );

      if (searchResult.answer && !result.agentAnswer) {
        result.agentAnswer = searchResult.answer;
      }
      steps = setStep(steps, 'analyzing', 'done');
    } catch (e) {
      if (!signal.aborted) {
        steps = setStep(steps, 'analyzing', 'error', String(e));
        log.error('Agent search failed:', e);
      }
    }
    emit();

    callbacks.onResult({ ...result });

  } catch (e) {
    if (signal.aborted) return;
    const msg = e instanceof Error ? e.message : String(e);
    log.error('Pipeline error:', e);
    callbacks.onError(msg);
  }
}
