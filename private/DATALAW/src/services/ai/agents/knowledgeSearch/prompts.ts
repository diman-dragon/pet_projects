import { normalizeTsvText } from '../../../rag/textExtractor/normalizeTsvText';
import type { SearchResult } from '../../../rag/UnifiedSearchService';
import type { RetrievalCandidate } from '../../../rag/hybridRetrievalService';
import {
  CONVERSATIONAL_IDENTITY_PROMPT,
  LAILA_ANTI_HALLUCINATION_RULES,
  LAILA_IDENTITY_HEADER,
} from '../../prompts/laila.identity';
import { buildLanguagePrompt } from './ollama';

const MAX_RESULTS = 8;
// FIX: унифицированный лимит на чанк для обоих путей (legacy + grounded)
const CHUNK_MAX_CHARS = 1000;

function formatChunkHeader(index: number, title: string, articleRef?: string | null): string {
  const ref = articleRef ? ` | ${articleRef}` : '';
  return `[${index}] ${title}${ref}`;
}

const CITATION_RULES = `
ПРАВИЛА ОТВЕТА (СТРОГО):
1. Используй ТОЛЬКО информацию из документов ниже. Никаких внешних знаний.
2. Если документы содержат ответ — дай краткий вывод, затем обоснование со ссылками [N].
3. Если в документах есть применимая норма (статья, пункт) — называй её и ставь [N].
4. Каждый юридический факт или вывод подтверждай ссылкой [N] в тексте.
5. НЕ ВЫВОДИ список источников, библиографию или раздел "Источники" в конце ответа.
   Ссылки [N] ставь ТОЛЬКО внутри текста, не отдельными строками.
6. НЕ ПИШИ названия файлов отдельными строками.
7. Если в документах НЕТ достаточной информации по вопросу — напиши РОВНО одно предложение:
   "В документах нет информации по этому вопросу."
   После этого — СТОП. Никаких общих рассуждений, советов, уточнений, продолжений.
8. Не копируй фрагменты дословно — пересказывай кратко своими словами.`;

export function buildLegacyAnalysisRequest(query: string, results: SearchResult[], queryLang?: string) {
  const systemPrompt = buildLanguagePrompt(
    queryLang,
    `${LAILA_IDENTITY_HEADER}

Ты — юридический аналитик базы знаний.
${CITATION_RULES}`,
  );

  const contextBlock = results
    .slice(0, MAX_RESULTS)
    .map((result, index) => {
      const cleanText = normalizeTsvText(result.chunkText).slice(0, CHUNK_MAX_CHARS);
      return `${formatChunkHeader(index + 1, result.title, result.articleRef)}\n${cleanText}`;
    })
    .join('\n\n---\n\n');

  return {
    systemPrompt,
    userMessage: `# ДОКУМЕНТЫ\n\n${contextBlock}\n\n# ВОПРОС\n\n${query}`,
  };
}

export function buildConversationalPrompt(queryLang?: string): string {
  return buildLanguagePrompt(queryLang, CONVERSATIONAL_IDENTITY_PROMPT);
}

export function buildGroundedAnswerPrompt(_chunks: RetrievalCandidate[], queryLang?: string): string {
  return buildLanguagePrompt(
    queryLang,
    `${LAILA_IDENTITY_HEADER}

${LAILA_ANTI_HALLUCINATION_RULES}

Ты отвечаешь как юридический аналитик: сначала ясный вывод по существу, затем краткое обоснование из документов.
${CITATION_RULES}`,
  );
}

export function buildGroundedAnswerUserMessage(chunks: RetrievalCandidate[], query: string): string {
  // FIX: нормализуем TSV/CSV текст и применяем лимит символов —
  // иначе сырые табы/точки-с-запятой идут в LLM, а большой chunkText
  // может переполнить контекстное окно (num_ctx=8192).
  const context = chunks
    .map((chunk, index) => {
      const cleanText = normalizeTsvText(chunk.chunkText).slice(0, CHUNK_MAX_CHARS);
      return `${formatChunkHeader(index + 1, chunk.title, chunk.articleRef)}\n${cleanText}`;
    })
    .join('\n\n---\n\n');

  return `# ДОКУМЕНТЫ\n\n${context}\n\n# ВОПРОС\n\n${query}`;
}
