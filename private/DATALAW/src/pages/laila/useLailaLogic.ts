// src/pages/laila/useLailaLogic.ts
// UPDATED: добавлены sessionId и loadSession для истории чатов

import { useState, useRef, useCallback, useMemo } from 'react';
import { LailaMode } from './types';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { useAIStatus } from './hooks/useAIStatus';
import { useChatStream } from './hooks/useChatStream';
import { useAnalysis } from './hooks/useAnalysis';
import { useAutoScroll } from './hooks/useAutoScroll';
import { WIDGET_HISTORY_LIMIT } from '../../services/ai/config';
import type { ChatMessage } from './types';

export function useLailaLogic() {
  const { t } = useI18n();
  const [mode, setMode] = useState<LailaMode>('chat');
  const [input, setInput] = useState('');

  const bottomRef = useRef<HTMLDivElement>(null);

  const welcomeMessage = useMemo(
    () => ({
      id: 'welcome',
      role: 'laila' as const,
      text: t('laila.chat.welcome'),
      timestamp: new Date(),
    }),
    [t],
  );

  const { aiReady, recheckAI } = useAIStatus();

  const {
    messages,
    setMessages,
    isThinking,
    send,
    stop: stopGeneration,
    clear,
    sessionId,
  } = useChatStream([welcomeMessage], {
    modelAgentId: 'widget',
    assistantId: 'laila',
    origin: 'chat',
    contextScope: 'laila:chat',
    historyLimit: WIDGET_HISTORY_LIMIT,
    sessionStorageKey: 'Dorlane:laila:session:chat',
    onAIStatusChange: (online) => {
      if (!online) recheckAI();
    },
  });

  const analysis = useAnalysis();

  useAutoScroll(bottomRef as any, [messages, isThinking]);

  const sendMessage = useCallback(
    (text?: string) => {
      const value = text ?? input;
      if (value) {
        send(value);
        setInput('');
      }
    },
    [input, send],
  );

  const clearChat = useCallback(() => {
    clear([welcomeMessage]);
    setInput('');
  }, [clear, welcomeMessage]);

  const clearAnalysis = useCallback(() => {
    analysis.reset();
  }, [analysis]);

  // Загружает сообщения из прошлой сессии в текущий вид (только просмотр)
  const loadSession = useCallback(
    (loadedMessages: ChatMessage[]) => {
      setMessages(loadedMessages);
    },
    [setMessages],
  );

  return {
    mode,
    setMode,
    aiReady,
    messages,
    input,
    setInput,
    isThinking,
    sendMessage,
    stopGeneration,
    clearChat,
    clearAnalysis,
    bottomRef,
    sessionId,
    loadSession,
    ...analysis,
  };
}
