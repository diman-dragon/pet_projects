import { getActiveChatModel } from '../../config';
import type { RetrievalCandidate } from '../../../rag/hybridRetrievalService';
import { buildConnectionFailureAnswer, callOllamaStream } from './ollama';
import { buildConversationalPrompt, buildGroundedAnswerPrompt, buildGroundedAnswerUserMessage } from './prompts';
import type { HistoryMessage } from './types';

export async function handleConversationalQuery(
  query: string,
  onToken: (token: string) => void,
  signal?: AbortSignal,
  queryLang?: string,
  history: HistoryMessage[] = [],
): Promise<string> {
  const model = await getActiveChatModel();
  const systemPrompt = buildConversationalPrompt(queryLang);

  return callOllamaStream({
    model,
    systemPrompt,
    userMessage: query,
    onToken,
    signal,
    queryLang,
    history,
  });
}

export async function generateGroundedAnswer(
  query: string,
  chunks: RetrievalCandidate[],
  onToken: (token: string) => void,
  signal?: AbortSignal,
  queryLang?: string,
  history: HistoryMessage[] = [],
): Promise<string> {
  const model = await getActiveChatModel();
  const systemPrompt = buildGroundedAnswerPrompt(chunks, queryLang);
  const userMessage = buildGroundedAnswerUserMessage(chunks, query);

  return callOllamaStream({
    model,
    systemPrompt,
    userMessage,
    onToken,
    signal,
    queryLang,
    history,
  });
}

// FIX: async — buildConnectionFailureAnswer теперь async (читает AppLanguageService + i18n).
// KnowledgeSearchAgent.produceAnswer должен await этот вызов.
export async function buildFallbackAnswer(
  error: unknown,
  chunkTitles: string[],
): Promise<string> {
  const errorMessage = error instanceof Error ? error.message : String(error);
  if (errorMessage.includes('Ollama connection failed')) {
    // queryLang убран — язык ошибки определяется из AppLanguageService внутри
    return buildConnectionFailureAnswer(chunkTitles);
  }
  return `Ошибка при генерации ответа: ${errorMessage}`;
}
