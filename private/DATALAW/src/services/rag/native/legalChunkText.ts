import { invokeRagCommand } from './invokeRagCommand';
import type { NativeTextChunk } from './types';

export interface LegalChunkOptions {
  docTitle: string;
  maxChunkChars?: number;
  minChunkChars?: number;
  childSize?: number;
}

export interface LegalChunkResult {
  chunks: NativeTextChunk[];
  /** Тип документа, определённый Rust-детектором. Пустая строка = не распознан. */
  docType: string;
}

/**
 * Вызывает Rust-чанкер для юридических документов.
 * Возвращает { chunks: [], docType: '' } если документ не распознан —
 * тогда вызывающий код должен упасть на rag_chunk_text.
 */
export async function legalChunkText(
  text: string,
  options: LegalChunkOptions,
): Promise<LegalChunkResult> {
  const raw = await invokeRagCommand<Array<NativeTextChunk & { docType: string }>>(
    'rag_chunk_legal',
    {
      request: {
        text,
        docTitle: options.docTitle,
        maxChunkChars: options.maxChunkChars,
        minChunkChars: options.minChunkChars,
        childSize: options.childSize,
      },
    },
  );

  if (!raw || raw.length === 0) {
    return { chunks: [], docType: '' };
  }

  const docType = raw[0]?.docType ?? '';
  const chunks: NativeTextChunk[] = raw.map(({ docType: _dt, ...rest }) => rest);
  return { chunks, docType };
}
