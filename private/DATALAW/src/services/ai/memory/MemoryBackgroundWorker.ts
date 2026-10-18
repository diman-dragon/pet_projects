import { EmbeddingService } from '../../rag/EmbeddingService';
import { createLogger } from '../../logger/Logger';
import { getMemoryEmbedModel, getMemoryConstants } from './constants';
import { ChatMemoryService } from './ChatMemoryService';
import { useOllamaStore } from '../../../store/ollamaStore';
import type { MemoryAssistantId, MemoryOrigin } from './types';
import { ConversationCompressor } from './ConversationCompressor';

const log = createLogger('MemoryBackgroundWorker');

type Job = () => Promise<void>;

class MemoryBackgroundWorkerClass {
  private readonly queue: Job[] = [];
  private running = false;

  schedule(job: Job): void {
    this.queue.push(job);
    if (!this.running) {
      setTimeout(() => this.drain().catch((error) => log.error('Worker queue failed', error)), 0);
    }
  }

  private async drain(): Promise<void> {
    this.running = true;
    try {
      while (this.queue.length > 0) {
        const next = this.queue.shift();
        if (!next) continue;

        try {
          await next();
        } catch (error) {
          log.error('Background memory job failed', error);
        }
      }
    } finally {
      this.running = false;
      // [Bug #7] Race condition: если новые задачи добавились во время drain — перезапускаем
      if (this.queue.length > 0) {
        setTimeout(() => this.drain().catch((error) => log.error('Worker queue failed', error)), 0);
      }
    }
  }

  scheduleEmbedding(params: {
    sessionId: string;
    agentId: MemoryAssistantId;
    origin: MemoryOrigin;
    contextScope: string;
    sourceType: 'message' | 'summary';
    sourceRef: string;
    role: 'user' | 'assistant' | null;
    text: string;
  }): void {
    this.schedule(async () => {
      const embedModel = await getMemoryEmbedModel();

      const recordId = await ChatMemoryService.saveMemoryRecord({
        ...params,
        embeddingModel: embedModel,
      });

      if (!recordId || !params.text.trim()) return;

      const embedding = await EmbeddingService.embed(params.text, embedModel, false);
      await ChatMemoryService.saveEmbedding({
        recordId: Number(recordId),
        sessionId: params.sessionId,
        agentId: params.agentId,
        model: embedModel,
        dim: embedding.dim,
        vectorJson: JSON.stringify(Array.from(embedding.vector)),
      });
    });
  }

  scheduleAutoSummary(
    sessionId: string,
    agentId: MemoryAssistantId,
    origin: MemoryOrigin,
    contextScope: string,
  ): void {
    this.schedule(async () => {
      // Получаем константы с учетом профиля машины
      const machineProfile = useOllamaStore.getState().machineProfile;
      const memoryConstants = getMemoryConstants(machineProfile);

      const pending = await ChatMemoryService.listMessagesForSummary(sessionId);
      if (pending.length <= memoryConstants.SUMMARY_SAFE_TAIL_MESSAGES) return;

      const totalTokens = pending.reduce((sum, item) => sum + item.token_estimate, 0);
      if (totalTokens < memoryConstants.SUMMARY_TRIGGER_TOKENS) return;

      const summarizable = pending.slice(
        0,
        Math.max(0, pending.length - memoryConstants.SUMMARY_SAFE_TAIL_MESSAGES),
      );
      if (summarizable.length === 0) return;

      const summaryText = await ConversationCompressor.compress(summarizable);
      const firstIndex = summarizable[0].message_index;
      const lastIndex = summarizable[summarizable.length - 1].message_index;

      const summaryId = await ChatMemoryService.saveSummary({
        sessionId,
        agentId,
        origin,
        contextScope,
        startMessageIndex: firstIndex,
        endMessageIndex: lastIndex,
        summaryText,
      });

      if (summaryId == null) return;

      this.scheduleEmbedding({
        sessionId,
        agentId,
        origin,
        contextScope,
        sourceType: 'summary',
        sourceRef: String(summaryId),
        role: null,
        text: summaryText,
      });
    });
  }
}

export const MemoryBackgroundWorker = new MemoryBackgroundWorkerClass();
