import { chunkText } from './native/chunkText';
import { DEFAULT_RAG_CONFIG } from './config';

export interface TextChunk {
  text: string;
  chunkIndex: number;
  charStart: number;
  charEnd: number;
  parentIndex?: number;
  articleRef?: string;
}

export const ChunkService = {
  chunk(text: string): Promise<TextChunk[]> {
    const { chunking } = DEFAULT_RAG_CONFIG;
    return chunkText(text, {
      chunkSize: chunking.chunkSize,
      overlapSize: chunking.overlapSize,
      minChunkSize: chunking.minChunkSize,
      maxChunkChars: chunking.maxChunkChars,
      childSize: chunking.childSize,
    });
  },

  estimateChunkCount(textLength: number): number {
    const { chunking } = DEFAULT_RAG_CONFIG;
    return Math.max(1, Math.ceil(textLength / (chunking.chunkSize - chunking.overlapSize)));
  },
};
