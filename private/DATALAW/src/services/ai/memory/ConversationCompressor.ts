import { getActiveChatModel } from '../config';
import { aiApi } from '../../backend/aiApi';
import { createLogger } from '../../logger/Logger';
import { useOllamaStore } from '../../../store/ollamaStore';
import { canRunCompressor, getSTMThreshold, getContextParams } from '../ModelContextProfile';
import type { ChatMessageRecord } from './types';

const log = createLogger('ConversationCompressor');

export const ConversationCompressor = {
  /**
   * [3.3] Проверяет, можно ли запустить компрессор на текущем профиле машины.
   * На minimal — слишком дорого, просто обрезаем историю.
   */
  canRun(): boolean {
    return canRunCompressor();
  },

  /**
   * [3.3] Максимальное число сообщений в STM перед запуском компрессора.
   */
  getThreshold(): number {
    return getSTMThreshold();
  },

  /**
   * Compresses a block of messages into a concise summary that preserves intent,
   * decisions, and context for future LLM consumption.
   */
  async compress(messages: ChatMessageRecord[]): Promise<string> {
    if (messages.length === 0) return '';

    // [3.3] Не запускаем на minimal-профиле
    if (!canRunCompressor()) {
      log.warn('Compressor skipped: minimal profile, truncating history instead');
      return `[История обрезана: ${messages.length} сообщений]`;
    }

    const dialogue = messages
      .map(
        (m) =>
          `${m.role === 'user' ? 'User' : 'Assistant'} #${m.message_index}: ${m.content}`,
      )
      .join('\n');

    const model = useOllamaStore.getState().selectedModel || (await getActiveChatModel());
    const ctxParams = getContextParams('summarize');

    try {
      const response = await aiApi.chat(
        model,
        [
          {
            role: 'system',
            content:
              'Ты — специализированный компрессор диалогов для ИИ-ассистента Лайла. ' +
              'Твоя задача — преобразовать блок разговора в плотное, информативное резюме. ' +
              'КРИТИЧЕСКИ ВАЖНО: Сохрани все намерения пользователя, принятые решения, важные сущности (имена, даты, ID), ' +
              'а также любые конкретные ограничения или инструкции, данные пользователем. ' +
              'НЕ включай разговорные шаблоны и пустые фразы. ' +
              'Выводи результат в виде маркированного списка фактов и решений.',
          },
          {
            role: 'user',
            content:
              `Сожми следующий блок разговора, сохранив всю ключевую информацию для долгосрочного контекста.\n\n` +
              `${dialogue}`,
          },
        ],
        false,
        ctxParams,
      );

      return response.content.trim();
    } catch (error) {
      log.error('Compression failed', error);
      return `[Compression Failed] Summary of messages ${messages[0].message_index}-${messages[messages.length - 1].message_index}`;
    }
  },
};
