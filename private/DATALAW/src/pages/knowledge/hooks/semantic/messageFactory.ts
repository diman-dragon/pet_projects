import type { ChatMessage } from '../useKnowledgeHub.types';

export function createChatMessage(role: ChatMessage['role'], content: string): ChatMessage {
  return {
    id: Math.random().toString(36).slice(2, 10),
    role,
    content,
    timestamp: Date.now(),
  };
}
