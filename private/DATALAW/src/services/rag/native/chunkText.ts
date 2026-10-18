import type { NativeTextChunk } from './types';
import { invokeRagCommand } from './invokeRagCommand';

export interface ChunkTextOptions {
  chunkSize?: number;
  overlapSize?: number;
  minChunkSize?: number;
  maxChunkChars?: number;
  childSize?: number;
}

export function chunkText(text: string, options?: ChunkTextOptions): Promise<NativeTextChunk[]> {
  return invokeRagCommand<NativeTextChunk[]>('rag_chunk_text', {
    request: {
      text,
      chunkSize: options?.chunkSize,
      overlapSize: options?.overlapSize,
      minChunkSize: options?.minChunkSize,
      maxChunkChars: options?.maxChunkChars,
      childSize: options?.childSize,
    },
  });
}
