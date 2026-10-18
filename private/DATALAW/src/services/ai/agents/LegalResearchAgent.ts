// src/services/ai/agents/LegalResearchAgent.ts
// RAG-агент для модуля «База знаний»
//
// ИСПРАВЛЕННЫЙ ПОТОК:
// 1. Получаем запрос пользователя
// 2. Агент генерирует суммари-расширение запроса (для лучшей векторизации)
// 3. Суммари векторизуется → поиск чанков
// 4. Агент получает: оригинальный запрос + суммари + чанки → финальный ответ
// 5. Каждая ссылка [N] встречается в тексте только по одному разу на факт
//
// FIX: добавлен лимит 1000 символов на чанк в contextBlock.
// Без лимита большие документы переполняли контекстное окно (num_ctx=8192).

import { aiApi } from '../../backend/aiApi';
import { getActiveChatModel } from '../config';
import { AppLanguageService } from '../../config/AppLanguageService';
import { getContextParams } from '../ModelContextProfile';
import { KNOWLEDGE_EMPTY_CONTEXT_MESSAGE } from '../prompts/knowledge.prompt';
import { createLogger } from '../../logger/Logger';
import { normalizeTsvText } from '../../rag/textExtractor/normalizeTsvText';

const log = createLogger('LegalResearchAgent');

export interface LegalResearchResult {
  answer: string;
  sources: Array<{ title: string; score: number; excerpt: string }>;
  isEmpty: boolean;
  confidence: number;
}

const MIN_RELEVANCE_SCORE = 0.2;
// FIX: лимит символов на чанк — соответствует CHUNK_MAX_CHARS в prompts.ts
const CHUNK_MAX_CHARS = 1000;

const QUERY_SUMMARY_PROMPT = `Ты — юридический аналитик. Твоя задача — расширить запрос пользователя
для улучшения семантического поиска по базе правовых документов.

Сгенерируй краткое суммари (2-4 предложения) которое:
- Раскрывает юридический смысл запроса
- Включает синонимы и смежные правовые термины
- Помогает найти релевантные фрагменты в базе

Отвечай ТОЛЬКО суммари, без пояснений.`;

const LEGAL_RESEARCH_CORE_PROMPT = `
Ты — юридический ассистент Laila. Твоя задача — отвечать на вопросы пользователя на основе найденных документов из базы знаний.

=== ИНСТРУКЦИЯ ===
Ты получаешь:
1. ОРИГИНАЛЬНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ — на него ты должен дать прямой ответ
2. СУММАРИ ЗАПРОСА — контекст для понимания юридического смысла
3. ИСТОРИЮ ДИАЛОГА (если есть) — для понимания контекста
4. НАЙДЕННЫЕ ДОКУМЕНТЫ — используй ТОЛЬКО их для ответа

=== ПРАВИЛА ОТВЕТА ===
1. Отвечай ТОЛЬКО на основе найденных документов. Не домышляй.
2. Начни с прямого ответа на вопрос пользователя.
3. Каждое утверждение подкрепляй ссылкой [N] на документ.
4. Структурируй сложные ответы: общее правило → частные случаи → практика.
5. Указывай конкретные нормы: статьи, пункты, как в документе.

=== ПРАВИЛА ЦИТИРОВАНИЯ ===
- Используй только номера найденных документов: [1], [2], [3]...
- Каждую ссылку [N] ставь ОДИН РАЗ рядом с утверждением которое она подтверждает
- Не повторяй одну и ту же ссылку несколько раз подряд
- Ссылки должны быть релевантны — не цитируй документ если он не относится к вопросу

=== ЕСЛИ ИНФОРМАЦИИ НЕДОСТАТОЧНО ===
Ответь только: [EMPTY_KB]

=== СТИЛЬ ===
- Профессиональный, для юриста
- Чёткий и структурированный
- Без воды и общих фраз
`;

export const LegalResearchAgent = {
  async generateQuerySummary(
    query: string,
    signal?: AbortSignal,
  ): Promise<string> {
    const model = await getActiveChatModel();

    try {
      const response = await aiApi.chat(
        model,
        [
          { role: 'system', content: QUERY_SUMMARY_PROMPT },
          { role: 'user', content: query },
        ],
        false,
        { num_ctx: 2048, num_predict: 256, temperature: 0.1 },
      );

      if (signal?.aborted) throw new Error('Aborted');

      const summary = response.content?.trim();
      if (summary && summary.length > 10) {
        log.info(`[Agent] Query summary generated (${summary.length} chars)`);
        return summary;
      }
    } catch (err) {
      if (signal?.aborted) throw err;
      log.warn('[Agent] Summary generation failed, using raw query:', err);
    }

    return query;
  },

  async answerFromChunks(
    query: string,
    querySummary: string,
    chunks: Array<{ title: string; chunkText: string; score: number; matchType: string }>,
    onToken?: (token: string) => void,
    signal?: AbortSignal,
    memoryContext?: string,
  ): Promise<LegalResearchResult> {
    const relevantChunks = chunks.filter((item) => {
      const pass = item.score >= MIN_RELEVANCE_SCORE;
      if (!pass) log.info(`[Agent] Filtered out: "${item.title.slice(0, 30)}" score=${item.score.toFixed(3)}`);
      return pass;
    });

    log.info(`[Agent] Chunks after filter: ${relevantChunks.length}/${chunks.length}`);

    if (!relevantChunks.length) {
      log.warn('[Agent] No relevant chunks — returning empty context message');
      if (onToken) {
        const msg = KNOWLEDGE_EMPTY_CONTEXT_MESSAGE;
        for (let i = 0; i < msg.length; i += 4) {
          if (signal?.aborted) break;
          onToken(msg.slice(i, i + 4));
          await new Promise((r) => setTimeout(r, 8));
        }
      }
      return { answer: KNOWLEDGE_EMPTY_CONTEXT_MESSAGE, sources: [], isEmpty: true, confidence: 0 };
    }

    const contextBlock = relevantChunks
      .map((item, index) => {
        const matchEmoji =
          item.matchType === 'hybrid' ? '⚡' : item.matchType === 'vector' ? '🔢' : '🔤';
        // FIX: нормализуем TSV/CSV + лимит символов на чанк.
        // Без лимита несколько больших чанков переполняют num_ctx=8192.
        const cleanText = normalizeTsvText(item.chunkText).slice(0, CHUNK_MAX_CHARS);
        return `[${index + 1}] ${matchEmoji} ${item.title} (релевантность: ${item.score.toFixed(2)})\n${cleanText}`;
      })
      .join('\n\n---\n\n');

    const model = await getActiveChatModel();
    const langInstruction = await AppLanguageService.getPromptInstruction();

    const contextSections = [langInstruction, LEGAL_RESEARCH_CORE_PROMPT];

    if (memoryContext?.trim()) {
      contextSections.push(`=== ИСТОРИЯ ДИАЛОГА ===\n${memoryContext}\n`);
    }

    if (querySummary && querySummary !== query) {
      contextSections.push(`=== КОНТЕКСТ ЗАПРОСА ===\n${querySummary}\n`);
    }

    contextSections.push(`=== НАЙДЕННЫЕ ДОКУМЕНТЫ ===\n${contextBlock}\n`);
    contextSections.push(
      `=== ЗАДАЧА ===\nОтветь на вопрос пользователя, используя ТОЛЬКО найденные документы. ` +
      `Каждое утверждение должно иметь ссылку [N] на источник. Каждую ссылку ставь один раз.`,
    );

    const systemContent = contextSections.join('\n\n');

    const messages = [
      { role: 'system' as const, content: systemContent },
      { role: 'user' as const, content: `ВОПРОС: ${query}\n\nДай развёрнутый ответ со ссылками на документы.` },
    ];

    if (signal?.aborted) throw new Error('Aborted');

    const response = await aiApi.chat(model, messages, false, getContextParams('rag'));

    if (signal?.aborted) throw new Error('Aborted');

    const fullText = response.content?.trim() ?? '';

    if (!fullText || fullText.includes('[EMPTY_KB]')) {
      return { answer: KNOWLEDGE_EMPTY_CONTEXT_MESSAGE, sources: [], isEmpty: true, confidence: 0 };
    }

    if (onToken) {
      const chunkSize = 4;
      for (let i = 0; i < fullText.length; i += chunkSize) {
        if (signal?.aborted) break;
        onToken(fullText.slice(i, i + chunkSize));
        if (i + chunkSize < fullText.length) {
          await new Promise((r) => setTimeout(r, 8));
        }
      }
    }

    const sources = relevantChunks.slice(0, 8).map((item) => ({
      title: item.title,
      score: item.score,
      excerpt: normalizeTsvText(item.chunkText).slice(0, 200) + (item.chunkText.length > 200 ? '…' : ''),
    }));

    const avgScore =
      relevantChunks.reduce((sum, item) => sum + item.score, 0) / relevantChunks.length;

    return {
      answer: fullText,
      sources,
      isEmpty: false,
      confidence: Math.min(avgScore, 1),
    };
  },
};
