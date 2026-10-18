// src/services/ai/config.ts
//
// FIX БАГ 5: убраны хардкоды AI_MODELS.widget и AI_MODELS.analyze.
// Причина: SetupWizard устанавливает фиксированные начальные модели
// gemma3:1b и bge-m3:latest, а агенты должны работать через selectedModel.
// Теперь единственный источник правды для chat-модели — ollamaStore.selectedModel.
//
import { DEFAULT_EMBEDDING_MODEL } from './embeddingModels';
import { useOllamaStore } from '@/store/ollamaStore';

// Для получения chat-модели в агентах используй:
//   import { useOllamaStore } from '@/store/ollamaStore';
//   const model = useOllamaStore.getState().selectedModel ?? DEFAULT_CHAT_MODEL;
//
// AI_MODELS.rag остаётся фиксированным — это embedding-модель, не chat.

// Fallback если selectedModel ещё не инициализирован в store
export const DEFAULT_CHAT_MODEL = 'gemma3:1b';

export const AI_MODELS = {
  // FIX: chat-модели убраны из хардкода.
  // Используй useOllamaStore.getState().selectedModel вместо AI_MODELS.widget / AI_MODELS.analyze.
  rag: DEFAULT_EMBEDDING_MODEL,
} as const;

// Вспомогательная функция для безопасного получения текущей chat-модели
// без прямого вызова хука (для использования вне React-компонентов)
export async function getActiveChatModel(): Promise<string> {
  try {
    // Lazy import чтобы избежать circular dependency
    return useOllamaStore.getState().selectedModel ?? DEFAULT_CHAT_MODEL;
  } catch {
    return DEFAULT_CHAT_MODEL;
  }
}

export const RAG_SIMILARITY_THRESHOLD = 0.25;
export const RAG_TOP_K = 5;
export type AgentId = keyof typeof AI_MODELS;
export const LOCAL_AI_HEALTH_TIMEOUT_MS = 4000;
export const WIDGET_HISTORY_LIMIT = 20;
