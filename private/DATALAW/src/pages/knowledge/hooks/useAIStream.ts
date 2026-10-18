import { useCallback, useRef, useState } from 'react';
import { OllamaService } from '../../../services/ai/OllamaService';
import { LegalResearchAgent } from '../../../services/ai/agents/LegalResearchAgent';
import { RagFacade } from '../../../services/rag/RagFacade';

type AgentId = 'widget' | 'analyze' | 'knowledge';
type AvailabilityState = 'ready' | 'offline' | 'timeout';

interface UseAIStreamOptions {
  agentId: AgentId;
}

const AI_FIRST_TOKEN_TIMEOUT_MS = 60000;

export function useAIStream({ agentId }: UseAIStreamOptions) {
  const [text, setText] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [done, setDone] = useState(false);
  const [availability, setAvailability] = useState<AvailabilityState>('ready');
  const abortRef = useRef<AbortController | null>(null);
  const timeoutRef = useRef<number | null>(null);

  const clearTimer = () => {
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  };

  const start = useCallback(
    async (prompt: string) => {
      abortRef.current?.abort();
      clearTimer();

      const controller = new AbortController();
      let timedOut = false;
      let firstTokenReceived = false;

      abortRef.current = controller;

      setText('');
      setDone(false);
      setAvailability('ready');
      setStreaming(true);

      timeoutRef.current = window.setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, AI_FIRST_TOKEN_TIMEOUT_MS);

      try {
        if (agentId === 'knowledge') {
          const querySummary = await LegalResearchAgent.generateQuerySummary(
            prompt,
            controller.signal,
          );
          const ragResults = await RagFacade.search({ query: prompt, topK: 6, hybrid: true });

          const result = await LegalResearchAgent.answerFromChunks(
            prompt,
            querySummary,
            ragResults,
            (token) => {
              if (!firstTokenReceived) {
                firstTokenReceived = true;
                clearTimer();
              }
              setText((prev) => prev + token);
            },
            controller.signal,
          );

          if (!firstTokenReceived) {
            clearTimer();
            setText(result.answer);
          }
        } else {
          const promptKey = (
            agentId === 'widget' ? 'WIDGET_SYSTEM_PROMPT' : 'ANALYZE_SYSTEM_PROMPT'
          ) as any;
          await OllamaService.streamChat(
            promptKey,
            prompt,
            (token) => {
              if (!firstTokenReceived) {
                firstTokenReceived = true;
                clearTimer();
              }
              setText((prev) => prev + token);
            },
            [],
            undefined,
          );
        }
      } catch (error: unknown) {
        if (error instanceof Error && error.name === 'AbortError') {
          if (timedOut) {
            setAvailability('timeout');
          }
          return;
        }

        console.error('[AIStream] Stream failed:', error);
        setAvailability('offline');
      } finally {
        clearTimer();
        setStreaming(false);
        setDone(true);
      }
    },
    [agentId],
  );

  const abort = useCallback(() => {
    clearTimer();
    abortRef.current?.abort();
    setStreaming(false);
  }, []);

  const reset = useCallback(() => {
    clearTimer();
    abortRef.current?.abort();
    setText('');
    setStreaming(false);
    setDone(false);
    setAvailability('ready');
  }, []);

  return {
    text,
    streaming,
    done,
    availability,
    offline: availability === 'offline',
    start,
    abort,
    reset,
  };
}
