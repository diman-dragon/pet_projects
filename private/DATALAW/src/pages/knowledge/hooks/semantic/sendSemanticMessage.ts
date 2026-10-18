import type { Dispatch, SetStateAction } from 'react';
import { translate } from '../../../../i18n/translations';
import { KnowledgeSearchAgent } from '../../../../services/ai/agents/KnowledgeSearchAgent';
import { AppLanguageService } from '../../../../services/config/AppLanguageService';
import type { ChatMessage, SemanticState } from '../useKnowledgeHub.types';
import { createChatMessage } from './messageFactory';
import { replaceMessageContent } from './semanticMessageUpdates';

interface SendSemanticMessageParams {
  userQuery: string;
  chatId: string;
  signal: AbortSignal;
  memoryMessages: Array<{ role: 'user' | 'assistant'; content: string }>;
  setSemanticState: Dispatch<SetStateAction<SemanticState>>;
}

export async function sendSemanticMessage({
  userQuery,
  chatId,
  signal,
  memoryMessages,
  setSemanticState,
}: SendSemanticMessageParams) {
  const thinkingMessage: ChatMessage = {
    ...createChatMessage('assistant', ''),
    sources: [],
  };

  setSemanticState((prev) => ({
    ...prev,
    messages: [...prev.messages, thinkingMessage],
    isThinking: false,
    isStreaming: true,
  }));

  let assistantContent = '';
  const queryLang = await AppLanguageService.get();
  const result = await KnowledgeSearchAgent.search(
    userQuery,
    (token) => {
      if (signal.aborted) {
        return;
      }

      assistantContent += token;
      setSemanticState((prev) => replaceMessageContent(prev, thinkingMessage.id, assistantContent));
    },
    signal,
    chatId,
    queryLang,
    memoryMessages,
  );

  if (signal.aborted) {
    throw new Error('Aborted');
  }

  setSemanticState((prev) => ({
    ...replaceMessageContent(prev, thinkingMessage.id, result.answer, result.sources),
    isStreaming: false,
    isThinking: false,
    abortController: null,
  }));

  return {
    assistantContent,
  };
}

export async function appendSemanticError(
  setSemanticState: Dispatch<SetStateAction<SemanticState>>,
) {
  const language = await AppLanguageService.get();
  const errorMessage = createChatMessage(
    'assistant',
    translate(language, 'knowledge.semantic.errorGeneric'),
  );

  setSemanticState((prev) => ({
    ...prev,
    messages: [...prev.messages, errorMessage],
    isStreaming: false,
    isThinking: false,
    abortController: null,
  }));
}
