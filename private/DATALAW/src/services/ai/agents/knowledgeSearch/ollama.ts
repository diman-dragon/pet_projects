import { getActiveChatModel } from '../../config';
import { getLangInstruction } from '../../prompts/laila.identity';
import { AppLanguageService } from '../../../config/AppLanguageService';
import { translate } from '../../../../i18n/translations';
import type { HistoryMessage } from './types';

const OLLAMA_API_URL = 'http://127.0.0.1:11434';
const RAG_NUM_CTX = 8192;
const RAG_NUM_PREDICT = 1024;
const RAG_TEMPERATURE = 0.15;

async function checkOllamaConnection(): Promise<boolean> {
  try {
    const response = await fetch(`${OLLAMA_API_URL}/api/tags`, {
      method: 'GET',
      signal: AbortSignal.timeout(2000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

// FIX: текст ошибки берётся из i18n (common.json errors.ollama.disconnected)
// через UI-язык (AppLanguageService), а не из queryLang.
// queryLang — язык запроса пользователя, не язык интерфейса.
// Ошибка подключения должна быть на языке UI, не на языке документа.
export async function getConnectionHelpText(): Promise<string> {
  const lang = await AppLanguageService.get();
  return translate(lang, 'errors.ollama.disconnected' as any);
}

export async function buildConnectionFailureAnswer(
  titles: string[],
): Promise<string> {
  const lang = await AppLanguageService.get();
  const help = translate(lang, 'errors.ollama.disconnected' as any);

  if (!titles.length) return help;

  const heading = translate(lang, 'errors.ollama.docsFound' as any);
  return `${help}\n\n${heading}\n${titles.map((title, index) => `${index + 1}. ${title}`).join('\n')}`;
}

export async function callOllamaStream(params: {
  systemPrompt: string;
  userMessage: string;
  onToken: (token: string) => void;
  signal?: AbortSignal;
  queryLang?: string;
  history?: HistoryMessage[];
  model?: string;
}): Promise<string> {
  const { systemPrompt, userMessage, onToken, signal, history = [] } = params;
  const model = params.model ?? (await getActiveChatModel());

  const isConnected = await checkOllamaConnection();
  if (!isConnected) {
    const errorMsg = await getConnectionHelpText();
    onToken(errorMsg);
    throw new Error(`Ollama connection failed: ${OLLAMA_API_URL}`);
  }

  // history already excludes the current user message — no slice needed
  const response = await fetch(`${OLLAMA_API_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({
      model,
      stream: true,
      options: {
        num_ctx: RAG_NUM_CTX,
        num_predict: RAG_NUM_PREDICT,
        temperature: RAG_TEMPERATURE,
      },
      messages: [
        { role: 'system', content: systemPrompt },
        ...history,
        { role: 'user', content: userMessage },
      ],
    }),
  });

  if (!response.ok) {
    throw new Error(`Ollama chat HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let fullText = '';

  while (true) {
    if (signal?.aborted) {
      break;
    }
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    const chunk = decoder.decode(value, { stream: true });
    for (const line of chunk.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      try {
        const json = JSON.parse(trimmed);
        const token = String(json?.message?.content ?? '');
        if (token) {
          fullText += token;
          onToken(token);
        }
        if (json?.done) {
          break;
        }
      } catch {
        // Ignore malformed stream chunks.
      }
    }
  }

  return fullText;
}

export function buildLanguagePrompt(queryLang?: string, basePrompt = ''): string {
  const langInstruction = getLangInstruction(queryLang);
  return `${langInstruction}\n\n${basePrompt}`.trim();
}
