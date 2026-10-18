import { ConnectionManager } from '../../db/ConnectionManager';
import type { LibraryIndexJobStatus } from './types';

export async function getLibraryCategories(): Promise<string[]> {
  const db = ConnectionManager.get('kb_main');
  const rows = await db.select<{ category: string }[]>(
    `SELECT DISTINCT category FROM library ORDER BY category`,
  );
  return rows.map((row) => row.category);
}

export async function getLibraryIndexJobStatus(
  libraryId: number,
): Promise<LibraryIndexJobStatus | null> {
  const db = ConnectionManager.get('kb_main');
  const rows = await db.select<
    {
      status: string;
      total_chunks: number;
      done_chunks: number;
      error_msg: string | null;
    }[]
  >(
    `SELECT status, total_chunks, done_chunks, error_msg
     FROM index_jobs
     WHERE library_id = $1
     ORDER BY created_at DESC
     LIMIT 1`,
    [libraryId],
  );

  if (rows.length === 0) {
    return null;
  }

  const [row] = rows;
  return {
    status: row.status,
    totalChunks: row.total_chunks,
    doneChunks: row.done_chunks,
    errorMsg: row.error_msg,
  };
}
