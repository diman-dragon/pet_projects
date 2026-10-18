import { ConnectionManager } from '../../db/ConnectionManager';
import { mapLibraryDocumentStatus } from './mappers';
import type { LibraryDocument, LibraryDocumentWithStatus } from './types';

export async function getAllLibraryDocuments(): Promise<LibraryDocumentWithStatus[]> {
  const db = ConnectionManager.get('kb_main');
  const rows = await db.select<
    (LibraryDocument & {
      chunk_count: number;
      embedding_count: number;
      last_job_status: string | null;
    })[]
  >(
    `SELECT
       l.*,
       COUNT(DISTINCT c.id)  AS chunk_count,
       COUNT(DISTINCT e.id)  AS embedding_count,
       (
         SELECT ij.status
         FROM   index_jobs ij
         WHERE  ij.library_id = l.id
         ORDER  BY ij.created_at DESC
         LIMIT  1
       ) AS last_job_status
     FROM library l
     LEFT JOIN chunks c ON c.library_id = l.id
     LEFT JOIN embeddings e ON e.library_id = l.id
     GROUP BY l.id
     ORDER BY l.created_at DESC`,
  );
  return rows.map(mapLibraryDocumentStatus);
}

export async function getLibraryDocumentById(id: number): Promise<LibraryDocumentWithStatus | null> {
  const db = ConnectionManager.get('kb_main');
  const rows = await db.select<LibraryDocument[]>(`SELECT * FROM library WHERE id = $1`, [id]);
  if (rows.length === 0) {
    return null;
  }

  const [document] = rows;
  const [{ cnt: chunkCount }] = await db.select<{ cnt: number }[]>(
    `SELECT COUNT(*) AS cnt FROM chunks WHERE library_id = $1`,
    [id],
  );
  const [{ cnt: embeddingCount }] = await db.select<{ cnt: number }[]>(
    `SELECT COUNT(*) AS cnt FROM embeddings WHERE library_id = $1`,
    [id],
  );

  return {
    ...document,
    chunk_count: chunkCount,
    embedding_count: embeddingCount,
    isIndexed: embeddingCount > 0,
    indexJobStatus: null,
  };
}

export async function searchLibraryByTitle(query: string): Promise<LibraryDocument[]> {
  const db = ConnectionManager.get('kb_main');
  const escaped = `%${query.replace(/'/g, "''")}%`;
  return db.select<LibraryDocument[]>(
    `SELECT * FROM library WHERE title LIKE $1 ORDER BY title LIMIT 50`,
    [escaped],
  );
}
