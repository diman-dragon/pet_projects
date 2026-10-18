import type { NativeSearchResult } from './types';
import { invokeRagCommand } from './invokeRagCommand';

// [2.3] Принимаем явные веса чтобы VectorSearchService мог передавать embeddingMode
export function mergeHybridResults(
  vectorResults: NativeSearchResult[],
  textResults: NativeSearchResult[],
  topK: number,
  cosineWeight?: number,
  keywordWeight?: number,
): Promise<NativeSearchResult[]> {
  return invokeRagCommand<NativeSearchResult[]>('rag_merge_hybrid', {
    request: {
      vectorResults,
      textResults,
      topK,
      cosineWeight,
      keywordWeight,
    },
  });
}
