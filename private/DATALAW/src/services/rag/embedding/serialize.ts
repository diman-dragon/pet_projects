// embedding/serialize.ts — сериализация векторов для SQLite
//
// Хранение: JSON TEXT "[0.1, 0.2, ...]" — совместимо с sqlite-vec и deserialization.
// Храним embedding как JSON, чтобы не зависеть от JS-специфики передачи BLOB.

export function serialize(vector: Float32Array): string {
  return JSON.stringify(Array.from(vector));
}

// Поддерживает: JSON-строка | number[] | Uint8Array | ArrayBuffer
export function deserialize(blob: unknown): Float32Array {
  if (!blob) return new Float32Array(0);
  try {
    if (typeof blob === 'string') {
      const parsed = JSON.parse(blob);
      if (Array.isArray(parsed)) return new Float32Array(parsed as number[]);
      if (typeof parsed === 'string') return new Float32Array(JSON.parse(parsed) as number[]);
      return new Float32Array(0);
    }
    if (Array.isArray(blob)) return new Float32Array(blob as number[]);
    const bytes = blob instanceof Uint8Array ? blob : new Uint8Array(blob as ArrayBuffer);
    if (bytes.length % 4 === 0)
      return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
  } catch (e) {
    console.error('deserialize error', e);
  }
  return new Float32Array(0);
}
