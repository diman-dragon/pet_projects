import type { SemanticState } from '../useKnowledgeHub.types';

export function createInitialSemanticState(): SemanticState {
  return {
    messages: [],
    isStreaming: false,
    isThinking: false,
    abortController: null,
    chatId: null,
  };
}

export function createSemanticChatId(): string {
  return `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}
