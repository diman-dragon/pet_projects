import type { SearchResult } from '../../../rag/UnifiedSearchService';
import { callOllamaStream } from './ollama';
import { MAX_RESULTS } from './pipelineShared';
import { buildLegacyAnalysisRequest } from './prompts';
import { getActiveChatModel } from '../../config';

export async function legacyAnalyze(
  query: string,
  results: SearchResult[],
  onToken: (token: string) => void,
  signal?: AbortSignal,
  queryLang?: string,
): Promise<string> {
  if (!results.length) {
    return '';
  }

  const model = await getActiveChatModel();
  const { systemPrompt, userMessage } = buildLegacyAnalysisRequest(
    query,
    results.slice(0, MAX_RESULTS),
    queryLang,
  );

  return callOllamaStream({
    model,
    systemPrompt,
    userMessage,
    onToken,
    signal,
    queryLang,
  });
}
