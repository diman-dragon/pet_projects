import type { ChatMessage, SemanticState } from '../useKnowledgeHub.types';

export function appendMessage(prev: SemanticState, message: ChatMessage): SemanticState {
  return {
    ...prev,
    messages: [...prev.messages, message],
  };
}

export function replaceMessageContent(
  prev: SemanticState,
  messageId: string,
  content: string,
  sources?: ChatMessage['sources'],
): SemanticState {
  return {
    ...prev,
    messages: prev.messages.map((message) =>
      message.id === messageId
        ? { ...message, content, sources: sources ?? message.sources }
        : message,
    ),
  };
}
