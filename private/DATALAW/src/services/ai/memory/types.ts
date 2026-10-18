import type { ChatMessage as OllamaMessage } from '../../backend/aiApi';

export type MemoryAssistantId = 'laila' | 'knowledge';
export type MemoryOrigin = 'widget' | 'chat' | 'analysis' | 'global' | 'knowledge';
export type MemoryMessageRole = 'user' | 'assistant';
export type MemorySourceType = 'message' | 'summary';

export interface ChatSessionRecord {
  id: string;
  agent_id: MemoryAssistantId;
  origin: MemoryOrigin;
  context_scope: string;
  title: string | null;
  last_message_at: string | null;
  last_summarized_index: number;
  created_at: string;
  updated_at: string;
}

export interface ChatMessageRecord {
  id: string;
  session_id: string;
  agent_id: MemoryAssistantId;
  message_index: number;
  role: MemoryMessageRole;
  content: string;
  token_estimate: number;
  created_at: string;
}

export interface ChatSummaryRecord {
  id: number;
  session_id: string;
  agent_id: MemoryAssistantId;
  origin: MemoryOrigin;
  context_scope: string;
  start_message_index: number;
  end_message_index: number;
  summary_text: string;
  token_estimate: number;
  created_at: string;
  updated_at: string;
}

export interface MemorySearchResult {
  recordId: number;
  sessionId: string;
  agentId: MemoryAssistantId;
  origin: MemoryOrigin;
  contextScope: string;
  sourceType: MemorySourceType;
  role: MemoryMessageRole | null;
  text: string;
  score: number;
  createdAt: string;
}

export interface MemoryContextBundle {
  history: OllamaMessage[];
  systemContext: string;
}
