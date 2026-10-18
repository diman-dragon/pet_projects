import type { LibraryDocument, LibraryDocumentWithStatus } from './types';

export function mapLibraryDocumentStatus(
  row: LibraryDocument & {
    chunk_count: number;
    embedding_count: number;
    last_job_status: string | null;
  },
): LibraryDocumentWithStatus {
  return {
    ...row,
    chunk_count: row.chunk_count,
    embedding_count: row.embedding_count,
    isIndexed: (row.embedding_count ?? 0) > 0,
    indexJobStatus: (row.last_job_status ?? null) as LibraryDocumentWithStatus['indexJobStatus'],
  };
}
