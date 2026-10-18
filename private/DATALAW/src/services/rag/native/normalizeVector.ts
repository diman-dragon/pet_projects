import { invokeRagCommand } from './invokeRagCommand';

export async function normalizeVector(vector: Float32Array): Promise<Float32Array> {
  const normalized = await invokeRagCommand<number[]>('rag_normalize', {
    vector: Array.from(vector),
  });

  return new Float32Array(normalized);
}
