import { ChatMemoryService } from './ChatMemoryService';
import { MemoryBackgroundWorker } from './MemoryBackgroundWorker';
import { MemoryContextService } from './MemoryContextService';
import type { MemoryAssistantId, MemoryOrigin } from './types';

interface UnifiedContextParams {
  sessionId: string;
  assistantId: MemoryAssistantId;
  origin: MemoryOrigin;
  contextScope: string;
  userMessage: string;
}

interface UnifiedPersistTurnParams {
  sessionId: string;
  assistantId: MemoryAssistantId;
  origin: MemoryOrigin;
  contextScope: string;
  userMessageId: string;
  userMessage: string;
  assistantMessageId: string;
  assistantMessage: string;
}

export const UnifiedMemoryCore = {
  async ensureSession(params: {
    sessionId: string;
    assistantId: MemoryAssistantId;
    origin: MemoryOrigin;
    contextScope: string;
    title?: string;
  }) {
    await ChatMemoryService.ensureSession(
      params.sessionId,
      params.assistantId,
      params.origin,
      params.contextScope,
      params.title,
    );
  },

  async prepareContext(params: UnifiedContextParams) {
    await this.ensureSession(params);
    return MemoryContextService.buildContext({
      sessionId: params.sessionId,
      agentId: params.assistantId,
      origin: params.origin,
      contextScope: params.contextScope,
      userMessage: params.userMessage,
    });
  },

  async persistConversationTurn(params: UnifiedPersistTurnParams) {
    await this.ensureSession(params);

    const userRecord = await ChatMemoryService.appendMessage({
      id: params.userMessageId,
      sessionId: params.sessionId,
      agentId: params.assistantId,
      role: 'user',
      content: params.userMessage,
    });

    const assistantRecord = await ChatMemoryService.appendMessage({
      id: params.assistantMessageId,
      sessionId: params.sessionId,
      agentId: params.assistantId,
      role: 'assistant',
      content: params.assistantMessage,
    });

    if (userRecord) {
      MemoryBackgroundWorker.scheduleEmbedding({
        sessionId: params.sessionId,
        agentId: params.assistantId,
        origin: params.origin,
        contextScope: params.contextScope,
        sourceType: 'message',
        sourceRef: userRecord.id,
        role: 'user',
        text: userRecord.content,
      });
    }

    if (assistantRecord) {
      MemoryBackgroundWorker.scheduleEmbedding({
        sessionId: params.sessionId,
        agentId: params.assistantId,
        origin: params.origin,
        contextScope: params.contextScope,
        sourceType: 'message',
        sourceRef: assistantRecord.id,
        role: 'assistant',
        text: assistantRecord.content,
      });
    }

    MemoryBackgroundWorker.scheduleAutoSummary(
      params.sessionId,
      params.assistantId,
      params.origin,
      params.contextScope,
    );
  },

  async persistAnalysisSnapshot(params: {
    sessionId: string;
    assistantId: MemoryAssistantId;
    contextScope: string;
    analysisInputId: string;
    analysisInput: string;
    resultId: string;
    resultText: string;
  }) {
    return this.persistConversationTurn({
      sessionId: params.sessionId,
      assistantId: params.assistantId,
      origin: 'analysis',
      contextScope: params.contextScope,
      userMessageId: params.analysisInputId,
      userMessage: params.analysisInput,
      assistantMessageId: params.resultId,
      assistantMessage: params.resultText,
    });
  },
};
