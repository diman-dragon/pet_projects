// src/pages/knowledge/components/ai/LailaSearchAnswer.tsx
//
// Лайла получает УЖЕ НАЙДЕННЫЕ chunks и синтезирует из них связный ответ.
// Никакого повторного поиска — работает строго по переданным результатам.

import { useEffect, useRef, useState, useCallback } from 'react';
import { Sparkles, StopCircle } from 'lucide-react';
import { KnowledgeSearchAgent } from '../../../../services/ai/agents/KnowledgeSearchAgent';
import { AITextBody } from './AITextBody';
import type { SearchResult } from '../../../../services/rag/UnifiedSearchService';

interface LailaSearchAnswerProps {
  query: string;
  results: SearchResult[];
}

export function LailaSearchAnswer({ query, results }: LailaSearchAnswerProps) {
  const [text, setText] = useState('');
  const [streaming, setStreaming] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const prevKeyRef = useRef('');

  const run = useCallback(async (
    q: string,
    chunks: SearchResult[],
    signal: AbortSignal,
  ) => {
    setText('');
    setStreaming(true);

    try {
      await KnowledgeSearchAgent.analyze(
        q,
        chunks,
        (token) => {
          if (signal.aborted) return;
          setText((prev) => prev + token);
        },
        signal,
      );
    } catch (err: unknown) {
      if (signal.aborted) return;
      console.warn('[LailaSearchAnswer] error:', err);
    } finally {
      if (!signal.aborted) setStreaming(false);
    }
  }, []);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    setStreaming(false);
  }, []);

  useEffect(() => {
    const trimmed = query.trim();
    // Ключ — запрос + id первых 3 чанков (стабильный идентификатор набора)
    const key = `${trimmed}::${results.slice(0, 3).map(r => r.chunkId).join(',')}`;

    if (!trimmed || results.length === 0) {
      abortRef.current?.abort();
      setText('');
      setStreaming(false);
      prevKeyRef.current = '';
      return;
    }

    if (key === prevKeyRef.current) return;
    prevKeyRef.current = key;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    run(trimmed, results, controller.signal);

    return () => { controller.abort(); };
  }, [query, results, run]);

  if (!query.trim() || results.length === 0) return null;
  if (!streaming && !text) return null;

  return (
    <div className="mb-3 rounded-xl border border-[var(--app-accent)]/20 bg-gradient-to-b from-[var(--app-accent)]/[0.04] to-transparent shadow-sm transition-all duration-300">
      {/* Заголовок */}
      <div className="flex items-center gap-2 border-b border-[var(--app-accent)]/10 px-3.5 py-2.5">
        <div className="relative flex size-5 shrink-0 items-center justify-center">
          {streaming && (
            <div
              className="absolute inset-0 rounded-full bg-[var(--app-accent)]/20 animate-ping"
              style={{ animationDuration: '2s' }}
            />
          )}
          <div className="relative flex size-5 items-center justify-center rounded-full bg-[var(--app-accent)]/15">
            <Sparkles
              size={10}
              className={`text-[var(--app-accent)] ${streaming ? 'animate-pulse' : ''}`}
            />
          </div>
        </div>

        <span className="text-[length:var(--text-xs)] font-black uppercase tracking-widest text-[var(--app-accent)]/80">
          Laila
        </span>

        {streaming ? (
          <span className="ml-auto flex items-center gap-1.5">
            <span className="inline-flex gap-0.5">
              {[0, 150, 300].map((d) => (
                <span
                  key={d}
                  className="size-1 animate-bounce rounded-full bg-black/25"
                  style={{ animationDelay: `${d}ms` }}
                />
              ))}
            </span>
            <button
              type="button"
              onClick={abort}
              className="ml-1 text-black/20 hover:text-black/50 transition-colors"
              title="Остановить"
            >
              <StopCircle size={12} />
            </button>
          </span>
        ) : (
          <span className="ml-auto text-[length:var(--text-xs)] text-black/20">
            {results.length} фрагм.
          </span>
        )}
      </div>

      {/* Тело */}
      <div className="px-3.5 py-3">
        {text ? (
          <AITextBody text={text} streaming={streaming} />
        ) : (
          <div className="flex gap-1 py-1">
            {[0, 150, 300].map((d) => (
              <span
                key={d}
                className="size-1.5 animate-bounce rounded-full bg-[var(--app-accent)]/40"
                style={{ animationDelay: `${d}ms` }}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
