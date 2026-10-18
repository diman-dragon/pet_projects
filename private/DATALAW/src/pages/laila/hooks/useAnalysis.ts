import { useEffect, useMemo, useState } from 'react';
import { OllamaService } from '../../../services/ai/OllamaService';
import { UnifiedMemoryCore } from '../../../services/ai/memory/UnifiedMemoryCore';
import { AppLanguageService } from '../../../services/config/AppLanguageService';
import type { AnalysisResult } from '../types';
import { uid } from '../utils/uid';

const ANALYSIS_TIMEOUT_MS = 120_000;

const ANALYSIS_LANGUAGE_HINTS: Record<string, string> = {
  ru: 'Верни все строковые значения JSON на русском языке. Если пользователь явно просит другой язык, выполни его просьбу.',
  en: 'Return all JSON string values in English. If the user explicitly requests another language, follow that request.',
  sr: 'Vrati sve tekstualne vrednosti JSON-a na srpskom (latinica). Ako korisnik izričito traži drugi jezik, ispoštuj taj zahtev.',
};

interface AnalysisRuntimeState {
  analysisText: string;
  analysisResult: AnalysisResult | null;
  streamingText: string;
  isAnalyzing: boolean;
  sessionId: string;
}

interface AnalysisRuntime {
  state: AnalysisRuntimeState;
  listeners: Set<() => void>;
  abortRef: AbortController | null;
  timeoutRef: ReturnType<typeof setTimeout> | null;
  partialText: string;
}

const runtimes = new Map<string, AnalysisRuntime>();

function createRuntime(key: string): AnalysisRuntime {
  return {
    state: {
      analysisText: '',
      analysisResult: null,
      streamingText: '',
      isAnalyzing: false,
      sessionId: `laila-analysis-${key}-${uid()}`,
    },
    listeners: new Set(),
    abortRef: null,
    timeoutRef: null,
    partialText: '',
  };
}

function getOrCreateRuntime(key: string): AnalysisRuntime {
  const existing = runtimes.get(key);
  if (existing) return existing;
  const runtime = createRuntime(key);
  runtimes.set(key, runtime);
  return runtime;
}

function cloneState(state: AnalysisRuntimeState): AnalysisRuntimeState {
  return { ...state };
}

function notify(runtime: AnalysisRuntime) {
  runtime.listeners.forEach((listener) => listener());
}

function setRuntimeState(runtime: AnalysisRuntime, updater: (state: AnalysisRuntimeState) => AnalysisRuntimeState) {
  runtime.state = updater(runtime.state);
  notify(runtime);
}

function extractJson(raw: string): string {
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced) return fenced[1].trim();
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start !== -1 && end !== -1 && end > start) return raw.slice(start, end + 1);
  return raw.trim();
}

function normalizeConfidence(raw: unknown): number {
  const value = Number(raw);
  if (Number.isNaN(value)) return 0;
  return value <= 1 ? Math.round(value * 100) : Math.round(value);
}

function buildPartialResult(fullText: string): AnalysisResult {
  const summaryMatch = fullText.match(/"summary"\s*:\s*"([^"]{10,})"/);
  const summary = summaryMatch
    ? summaryMatch[1]
    : fullText.length > 20
      ? `${fullText.slice(0, 500)}${fullText.length > 500 ? '...' : ''}`
      : 'Анализ прерван: превышено время ожидания или модель вернула неполный ответ.';

  const risksMatch = fullText.match(/"risks"\s*:\s*\[([^\]]*)/);
  const risks: string[] = [];
  if (risksMatch) {
    const riskItems = risksMatch[1].matchAll(/"([^"]{5,})"/g);
    for (const match of riskItems) {
      risks.push(match[1]);
    }
  }

  return {
    summary,
    keyPoints: [],
    risks,
    articles: [],
    confidence: 0,
  };
}

async function runAnalysisInternal(runtime: AnalysisRuntime) {
  if (!runtime.state.analysisText.trim() || runtime.state.isAnalyzing) return;

  const analysisText = runtime.state.analysisText;
  const controller = new AbortController();

  runtime.abortRef = controller;
  runtime.partialText = '';

  setRuntimeState(runtime, (state) => ({
    ...state,
    isAnalyzing: true,
    analysisResult: null,
    streamingText: '',
  }));

  runtime.timeoutRef = setTimeout(() => {
    controller.abort();
    setRuntimeState(runtime, (state) => ({
      ...state,
      isAnalyzing: false,
      streamingText: '',
      analysisResult: buildPartialResult(runtime.partialText),
    }));
  }, ANALYSIS_TIMEOUT_MS);

  try {
    const prepared = await UnifiedMemoryCore.prepareContext({
      sessionId: runtime.state.sessionId,
      assistantId: 'laila',
      origin: 'analysis',
      contextScope: 'laila:analysis',
      userMessage: analysisText,
    });

    const responseLanguage = await AppLanguageService.get();
    const languageHint = ANALYSIS_LANGUAGE_HINTS[responseLanguage] ?? ANALYSIS_LANGUAGE_HINTS.ru;

    let fullText = '';

    await OllamaService.streamChat(
      'ANALYZE_SYSTEM_PROMPT',
      `${languageHint}\n\nДокумент для анализа:\n${analysisText}`,
      (token) => {
        fullText += token;
        runtime.partialText = fullText;
        setRuntimeState(runtime, (state) => ({
          ...state,
          streamingText: fullText,
        }));
      },
      prepared.history.map((message) => ({ role: message.role, content: message.content })),
      prepared.systemContext || undefined,
      undefined,
      responseLanguage,
    );

    if (runtime.timeoutRef) {
      clearTimeout(runtime.timeoutRef);
      runtime.timeoutRef = null;
    }

    if (controller.signal.aborted) return;

    let nextResult: AnalysisResult;
    try {
      const parsed = JSON.parse(extractJson(fullText));
      nextResult = {
        summary: typeof parsed.summary === 'string' ? parsed.summary : fullText,
        keyPoints: Array.isArray(parsed.keyPoints) ? parsed.keyPoints : [],
        risks: Array.isArray(parsed.risks) ? parsed.risks : [],
        articles: Array.isArray(parsed.articles) ? parsed.articles : [],
        confidence: normalizeConfidence(parsed.confidence),
      };
    } catch {
      nextResult = buildPartialResult(fullText);
    }

    setRuntimeState(runtime, (state) => ({
      ...state,
      analysisResult: nextResult,
    }));

    await UnifiedMemoryCore.persistAnalysisSnapshot({
      sessionId: runtime.state.sessionId,
      assistantId: 'laila',
      contextScope: 'laila:analysis',
      analysisInputId: uid(),
      analysisInput: analysisText,
      resultId: uid(),
      resultText: fullText,
    });
  } catch (error: any) {
    if (runtime.timeoutRef) {
      clearTimeout(runtime.timeoutRef);
      runtime.timeoutRef = null;
    }

    if (error?.name !== 'AbortError' && runtime.partialText) {
      setRuntimeState(runtime, (state) => ({
        ...state,
        analysisResult: buildPartialResult(runtime.partialText),
      }));
    }
  } finally {
    if (!controller.signal.aborted) {
      setRuntimeState(runtime, (state) => ({
        ...state,
        isAnalyzing: false,
        streamingText: '',
      }));
    }
    runtime.abortRef = null;
  }
}

function stopAnalysisInternal(runtime: AnalysisRuntime) {
  if (runtime.timeoutRef) {
    clearTimeout(runtime.timeoutRef);
    runtime.timeoutRef = null;
  }
  runtime.abortRef?.abort();
  setRuntimeState(runtime, (state) => ({
    ...state,
    isAnalyzing: false,
    streamingText: '',
    analysisResult: runtime.partialText ? buildPartialResult(runtime.partialText) : state.analysisResult,
  }));
}

function resetAnalysisInternal(runtime: AnalysisRuntime, key: string) {
  if (runtime.timeoutRef) {
    clearTimeout(runtime.timeoutRef);
    runtime.timeoutRef = null;
  }
  runtime.abortRef?.abort();
  runtime.partialText = '';
  setRuntimeState(runtime, () => ({
    analysisText: '',
    analysisResult: null,
    streamingText: '',
    isAnalyzing: false,
    sessionId: `laila-analysis-${key}-${uid()}`,
  }));
}

export function useAnalysis(scopeKey = 'laila-main') {
  const runtime = useMemo(() => getOrCreateRuntime(scopeKey), [scopeKey]);
  const [snapshot, setSnapshot] = useState<AnalysisRuntimeState>(() => cloneState(runtime.state));

  useEffect(() => {
    const sync = () => setSnapshot(cloneState(runtime.state));
    runtime.listeners.add(sync);
    sync();
    return () => {
      runtime.listeners.delete(sync);
    };
  }, [runtime]);

  return {
    analysisText: snapshot.analysisText,
    setAnalysisText: (value: string) =>
      setRuntimeState(runtime, (state) => ({
        ...state,
        analysisText: value,
      })),
    analysisResult: snapshot.analysisResult,
    streamingText: snapshot.streamingText,
    isAnalyzing: snapshot.isAnalyzing,
    runAnalysis: () => runAnalysisInternal(runtime),
    stopAnalysis: () => stopAnalysisInternal(runtime),
    reset: () => resetAnalysisInternal(runtime, scopeKey),
  };
}
