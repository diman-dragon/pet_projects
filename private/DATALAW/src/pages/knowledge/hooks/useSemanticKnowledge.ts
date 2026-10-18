import { useCallback, useState } from 'react';
import { searchLogger } from '../../../services/logger/SearchLogger';
import { createChatMessage } from './semantic/messageFactory';
import { appendMessage } from './semantic/semanticMessageUpdates';
import { appendSemanticError, sendSemanticMessage } from './semantic/sendSemanticMessage';
import { createInitialSemanticState, createSemanticChatId } from './semantic/semanticState';
import { useSemanticMemory } from './semantic/useSemanticMemory';

export function useSemanticKnowledge() {
  const [semanticState, setSemanticState] = useState(createInitialSemanticState);
  const { memoryMessages, memoryRef, addToMemory, clearMemory } = useSemanticMemory();

  const sendMessage = useCallback(
    async (userQuery: string) => {
      if (!userQuery.trim() || semanticState.isStreaming || semanticState.isThinking) {
        return;
      }

      const userMessage = createChatMessage('user', userQuery);
      const chatId = semanticState.chatId ?? createSemanticChatId();
      if (!semanticState.chatId) {
        searchLogger.startNewChat(chatId);
      }

      const abortController = new AbortController();
      setSemanticState((prev) => ({
        ...appendMessage(prev, userMessage),
        isThinking: true,
        abortController,
        chatId,
      }));
      addToMemory('user', userQuery);

      try {
        const { assistantContent } = await sendSemanticMessage({
          userQuery,
          chatId,
          signal: abortController.signal,
          memoryMessages: memoryRef.current,
          setSemanticState,
        });
        addToMemory('assistant', assistantContent);
      } catch (error) {
        if (abortController.signal.aborted) {
          setSemanticState((prev) => ({
            ...prev,
            isStreaming: false,
            isThinking: false,
            abortController: null,
          }));
          return;
        }

        console.error('[SEMANTIC] Error:', error);
        await appendSemanticError(setSemanticState);
      }
    },
    [addToMemory, memoryRef, semanticState.chatId, semanticState.isStreaming, semanticState.isThinking],
  );

  const abortSemantic = useCallback(() => {
    semanticState.abortController?.abort();
    setSemanticState((prev) => ({
      ...prev,
      isStreaming: false,
      isThinking: false,
      abortController: null,
    }));
  }, [semanticState.abortController]);

  const clearChat = useCallback(() => {
    semanticState.abortController?.abort();
    setSemanticState(createInitialSemanticState());
    clearMemory();
  }, [clearMemory, semanticState.abortController]);

  return {
    semanticState,
    sendMessage,
    abortSemantic,
    clearChat,
    memory: {
      messages: memoryMessages,
      add: addToMemory,
      clear: clearMemory,
    },
  };
}
