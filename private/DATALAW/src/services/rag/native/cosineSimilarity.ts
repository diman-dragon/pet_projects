import { invokeRagCommand } from './invokeRagCommand';

export function cosineSimilarity(a: Float32Array, b: Float32Array): Promise<number> {
  return invokeRagCommand<number>('rag_cosine_similarity', {
    a: Array.from(a),
    b: Array.from(b),
  });
}
