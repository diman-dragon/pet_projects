import { getActiveChatModel } from '../../ai/config';
import { detectLang } from '../../search/languageDetection';
import { OLLAMA_API_URL, TIMEOUT_MS, UNDERSTANDING_PROMPT } from './constants';
import { extractJson } from './helpers';
import type { PreprocessedQuery, QueryUnderstandingResult } from './types';

export async function requestRemoteUnderstanding(
  query: string,
  preprocessed: PreprocessedQuery,
): Promise<QueryUnderstandingResult> {
  const model = await getActiveChatModel();
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(`${OLLAMA_API_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: UNDERSTANDING_PROMPT.replace('{intentHint}', preprocessed.intentHint) },
          { role: 'user', content: preprocessed.normalized },
        ],
        stream: false,
        options: {
          temperature: 0,
          num_predict: 256,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();
    const content = String(result?.message?.content || '').trim();
    const parsed = extractJson(content);
    const expansions = Array.isArray(parsed.expansions)
      ? parsed.expansions.slice(0, 3).filter((entry: unknown) => typeof entry === 'string')
      : [];

    return {
      rewritten_query: typeof parsed.rewritten_query === 'string' && parsed.rewritten_query.length > 0
        ? parsed.rewritten_query
        : query,
      expansions,
      intent: ['factual', 'legal', 'procedural', 'biographical'].includes(parsed.intent)
        ? parsed.intent
        : 'factual',
      lang: detectLang(query, 'ru') ?? 'ru',
      hyde_snippet: typeof parsed.hyde_snippet === 'string' && parsed.hyde_snippet.length > 10
        ? parsed.hyde_snippet
        : undefined,
    };
  } finally {
    clearTimeout(timeoutId);
  }
}
