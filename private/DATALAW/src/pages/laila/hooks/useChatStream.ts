import { useEffect, useState } from 'react';
import type { SidecarStatus } from '../../../services/backend/aiApi';
import { LegalResearchAgent } from '../../../services/ai/agents/LegalResearchAgent';
import { UnifiedMemoryCore } from '../../../services/ai/memory/UnifiedMemoryCore';
import type { MemoryAssistantId, MemoryOrigin } from '../../../services/ai/memory/types';
import { OllamaService } from '../../../services/ai/OllamaService';
import { tryHandleLailaAction } from '../../../services/ai/LailaActionService';
import { createLogger } from '../../../services/logger/Logger';
import { RagFacade } from '../../../services/rag/RagFacade';
import { useOllamaStore } from '../../../store/ollamaStore';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { ChatMessage } from '../types';
import { uid } from '../utils/uid';
import { ChatMemoryService } from '../../../services/ai/memory/ChatMemoryService';

const COLD_START_TIMEOUT_MS = 60_000;
const log = createLogger('useChatStream');

type ModelAgentId = 'widget' | 'analyze' | 'knowledge';

const ROUTE_LABEL_KEYS: Record<string, string> = {
  dashboard: 'sidebar.menu.dashboard',
  clients: 'sidebar.menu.clients',
  users: 'sidebar.menu.users',
  knowledge: 'sidebar.menu.knowledge',
  laila: 'sidebar.menu.laila',
  systematizer: 'sidebar.menu.systematizer',
  settings: 'sidebar.menu.settings',
  audit: 'header.auditLog',
  trash: 'sidebar.menu.trash',
  'ai-control': 'sidebar.menu.ai_control',
};

interface UseChatStreamOptions {
  modelAgentId: ModelAgentId;
  assistantId: MemoryAssistantId;
  origin: MemoryOrigin;
  contextScope: string;
  historyLimit?: number;
  sessionStorageKey?: string;
  onAIStatusChange?: (online: boolean) => void;
}

interface RuntimeEnv {
  t: (key: any, options?: Record<string, string | number>) => string;
  refreshStatus: () => Promise<SidecarStatus>;
  onAIStatusChange?: (online: boolean) => void;
}

interface ChatRuntimeState {
  messages: ChatMessage[];
  isThinking: boolean;
  sessionId: string;
}

interface ChatRuntime {
  key: string;
  state: ChatRuntimeState;
  initialMessages: ChatMessage[];
  options: UseChatStreamOptions;
  env: RuntimeEnv | null;
  hydrated: boolean;
  hydrationPromise: Promise<void> | null;
  abortRef: AbortController | null;
  listeners: Set<() => void>;
}

const runtimes = new Map<string, ChatRuntime>();

function cloneState(state: ChatRuntimeState): ChatRuntimeState {
  return {
    sessionId: state.sessionId,
    isThinking: state.isThinking,
    messages: [...state.messages],
  };
}

function notify(runtime: ChatRuntime) {
  runtime.listeners.forEach((listener) => listener());
}

function setRuntimeState(runtime: ChatRuntime, updater: (state: ChatRuntimeState) => ChatRuntimeState) {
  runtime.state = updater(runtime.state);
  notify(runtime);
}

function getStorageSessionId(storageKey: string): string {
  const existing = typeof window !== 'undefined' ? window.localStorage.getItem(storageKey) : null;
  const sessionId = existing ?? uid();

  if (typeof window !== 'undefined' && !existing) {
    window.localStorage.setItem(storageKey, sessionId);
  }

  return sessionId;
}

function mapMemoryMessage(message: {
  id: string;
  role: string;
  content: string;
  created_at: string;
}): ChatMessage {
  return {
    id: message.id,
    role: message.role === 'user' ? 'user' : 'laila',
    text: message.content,
    timestamp: new Date(message.created_at),
  };
}

function updateAssistantMessage(runtime: ChatRuntime, streamId: string, text: string, isTyping = true) {
  setRuntimeState(runtime, (state) => ({
    ...state,
    messages: state.messages.map((message) =>
      message.id === streamId ? { ...message, text, isTyping } : message,
    ),
  }));
}

async function hydrateRuntime(runtime: ChatRuntime) {
  if (runtime.hydrated) return;
  if (runtime.hydrationPromise) {
    await runtime.hydrationPromise;
    return;
  }

  runtime.hydrationPromise = (async () => {
    await UnifiedMemoryCore.ensureSession({
      sessionId: runtime.state.sessionId,
      assistantId: runtime.options.assistantId,
      origin: runtime.options.origin,
      contextScope: runtime.options.contextScope,
    }).catch(() => undefined);

    const stored = await ChatMemoryService.listSessionMessages(runtime.state.sessionId).catch(() => []);
    if (stored.length > 0) {
      setRuntimeState(runtime, (state) => ({
        ...state,
        messages: stored.map(mapMemoryMessage),
      }));
    }

    runtime.hydrated = true;
  })().finally(() => {
    runtime.hydrationPromise = null;
  });

  await runtime.hydrationPromise;
}

function stopRuntime(runtime: ChatRuntime) {
  runtime.abortRef?.abort();
  runtime.abortRef = null;
  setRuntimeState(runtime, (state) => ({
    ...state,
    isThinking: false,
    messages: state.messages.map((message, index, items) =>
      index === items.length - 1 && message.role === 'laila' ? { ...message, isTyping: false } : message,
    ),
  }));
}

async function sendThroughRuntime(runtime: ChatRuntime, value: string) {
  const trimmed = value.trim();
  if (!trimmed || runtime.state.isThinking || !runtime.env) return;

  await hydrateRuntime(runtime);

  const { assistantId, contextScope, historyLimit, modelAgentId, origin } = runtime.options;
  const { onAIStatusChange, refreshStatus, t } = runtime.env;

  const localAction = await tryHandleLailaAction(trimmed);
  if (localAction.handled && localAction.route) {
    const labelKey = ROUTE_LABEL_KEYS[localAction.route] ?? 'sidebar.menu.dashboard';
    setRuntimeState(runtime, (state) => ({
      ...state,
      messages: [
        ...state.messages,
        { id: uid(), role: 'user', text: trimmed, timestamp: new Date() },
        {
          id: uid(),
          role: 'laila',
          text: t('laila.action.openedPage', {
            page: localAction.title ?? t(labelKey),
          }),
          timestamp: new Date(),
        },
      ],
    }));
    return;
  }

  const latestStatus = await refreshStatus();
  const online = latestStatus.running;
  onAIStatusChange?.(online);

  const userMsg: ChatMessage = { id: uid(), role: 'user', text: trimmed, timestamp: new Date() };
  setRuntimeState(runtime, (state) => ({
    ...state,
    isThinking: true,
    messages: [...state.messages, userMsg],
  }));

  if (!online) {
    setRuntimeState(runtime, (state) => ({
      ...state,
      isThinking: false,
      messages: [
        ...state.messages,
        {
          id: uid(),
          role: 'laila',
          text: t('laila.chat.ollamaOffline'),
          timestamp: new Date(),
        },
      ],
    }));
    return;
  }

  const streamId = uid();
  setRuntimeState(runtime, (state) => ({
    ...state,
    messages: [
      ...state.messages,
      { id: streamId, role: 'laila', text: '', timestamp: new Date(), isTyping: true },
    ],
  }));

  const controller = new AbortController();
  runtime.abortRef = controller;

  let accumulated = '';
  let streamCompleted = false;
  let firstTokenReceived = false;

  let coldStartTimer: ReturnType<typeof setTimeout> | null = setTimeout(() => {
    if (!firstTokenReceived) {
      log.warn('Cold start timeout: no first token in 60s, aborting');
      controller.abort();
    }
  }, COLD_START_TIMEOUT_MS);

  const clearColdTimer = () => {
    if (!coldStartTimer) return;
    clearTimeout(coldStartTimer);
    coldStartTimer = null;
  };

  try {
    const prepared = await UnifiedMemoryCore.prepareContext({
      sessionId: runtime.state.sessionId,
      assistantId,
      origin,
      contextScope,
      userMessage: trimmed,
    });

    const junkPatterns = [
      'В документах нет информации',
      'База знаний пуста',
      '[EMPTY_KB]',
      'laila.chat.error',
      'laila.chat.ollamaOffline',
    ];

    const history = (historyLimit ? prepared.history.slice(-historyLimit) : prepared.history)
      .filter((message) => {
        if (message.role !== 'assistant') return true;
        return !junkPatterns.some((pattern) => message.content.includes(pattern));
      })
      .map((message) => ({
        role: message.role,
        content: message.content,
      }));

    const handleToken = (token: string) => {
      if (!firstTokenReceived) {
        firstTokenReceived = true;
        clearColdTimer();
      }
      accumulated += token;
      updateAssistantMessage(runtime, streamId, accumulated, true);
    };

    if (modelAgentId === 'knowledge') {
      const querySummary = await LegalResearchAgent.generateQuerySummary(trimmed, controller.signal);
      const ragResults = await RagFacade.search({ query: trimmed, topK: 10, hybrid: true });
      const result = await LegalResearchAgent.answerFromChunks(
        trimmed,
        querySummary,
        ragResults,
        handleToken,
        controller.signal,
        prepared.systemContext || undefined,
      );

      if (!accumulated && result.answer) {
        accumulated = result.answer;
        updateAssistantMessage(runtime, streamId, accumulated, true);
      }
    } else {
      const promptKey: 'WIDGET_SYSTEM_PROMPT' | 'ANALYZE_SYSTEM_PROMPT' =
        modelAgentId === 'widget' ? 'WIDGET_SYSTEM_PROMPT' : 'ANALYZE_SYSTEM_PROMPT';

      await OllamaService.streamChat(
        promptKey,
        trimmed,
        handleToken,
        history,
        prepared.systemContext || undefined,
      );
    }

    streamCompleted = true;
    updateAssistantMessage(runtime, streamId, accumulated, false);
  } catch (error: any) {
    if (error?.name === 'AbortError') return;

    updateAssistantMessage(runtime, streamId, `${t('laila.chat.error')}: ${error?.message ?? error}`, false);
  } finally {
    clearColdTimer();
    runtime.abortRef = null;
    setRuntimeState(runtime, (state) => ({
      ...state,
      isThinking: false,
    }));
  }

  if (!streamCompleted) return;

  await UnifiedMemoryCore.persistConversationTurn({
    sessionId: runtime.state.sessionId,
    assistantId,
    origin,
    contextScope,
    userMessageId: userMsg.id,
    userMessage: trimmed,
    assistantMessageId: streamId,
    assistantMessage: accumulated,
  }).catch((error) => log.error('Memory persist failed', error));
}

function clearRuntime(runtime: ChatRuntime, resetTo: ChatMessage[]) {
  stopRuntime(runtime);
  const nextSessionId = uid();
  runtime.hydrated = true;

  if (typeof window !== 'undefined') {
    window.localStorage.setItem(runtime.key, nextSessionId);
  }

  setRuntimeState(runtime, () => ({
    sessionId: nextSessionId,
    isThinking: false,
    messages: resetTo,
  }));

  void UnifiedMemoryCore.ensureSession({
    sessionId: nextSessionId,
    assistantId: runtime.options.assistantId,
    origin: runtime.options.origin,
    contextScope: runtime.options.contextScope,
  }).catch(() => undefined);
}

function replaceMessages(runtime: ChatRuntime, messages: ChatMessage[]) {
  setRuntimeState(runtime, (state) => ({
    ...state,
    messages,
  }));
}

function getOrCreateRuntime(storageKey: string, initialMessages: ChatMessage[], options: UseChatStreamOptions) {
  const existing = runtimes.get(storageKey);
  if (existing) {
    existing.options = options;
    return existing;
  }

  const runtime: ChatRuntime = {
    key: storageKey,
    initialMessages,
    options,
    env: null,
    hydrated: false,
    hydrationPromise: null,
    abortRef: null,
    listeners: new Set(),
    state: {
      messages: initialMessages,
      isThinking: false,
      sessionId: getStorageSessionId(storageKey),
    },
  };

  runtimes.set(storageKey, runtime);
  return runtime;
}

export function useChatStream(initialMessages: ChatMessage[], options: UseChatStreamOptions) {
  const { t } = useI18n();
  const refreshStatus = useOllamaStore((state) => state.refreshStatus);

  const storageKey = options.sessionStorageKey ?? `Dorlane:laila:session:${options.origin}`;
  const runtime = getOrCreateRuntime(storageKey, initialMessages, options);

  runtime.env = { t, refreshStatus, onAIStatusChange: options.onAIStatusChange };
  runtime.initialMessages = initialMessages;

  const [snapshot, setSnapshot] = useState<ChatRuntimeState>(() => cloneState(runtime.state));

  useEffect(() => {
    const sync = () => setSnapshot(cloneState(runtime.state));
    runtime.listeners.add(sync);
    sync();
    return () => {
      runtime.listeners.delete(sync);
    };
  }, [runtime]);

  useEffect(() => {
    void hydrateRuntime(runtime);
  }, [runtime]);

  return {
    messages: snapshot.messages,
    setMessages: (messages: ChatMessage[]) => replaceMessages(runtime, messages),
    isThinking: snapshot.isThinking,
    send: (value: string) => sendThroughRuntime(runtime, value),
    stop: () => stopRuntime(runtime),
    clear: (resetTo: ChatMessage[]) => clearRuntime(runtime, resetTo),
    sessionId: snapshot.sessionId,
  };
}
