// src/services/rag/indexer/libraryUpsert.ts

// Версия чанкера — менять при изменении логики чанкинга для принудительной переиндексации
export const CHUNKER_VERSION = 1;

export interface LibraryUpsertResult {
  libraryId: number;
  isNew: boolean;
  needsReindex: boolean;
}

async function cleanupDocumentIndex(db: any, libraryId: number): Promise<void> {
  await db.execute(`DELETE FROM chunks WHERE library_id = $1`, [libraryId]);
  try {
    await db.execute(`DELETE FROM vec_embeddings WHERE library_id = $1`, [libraryId]);
  } catch {
    // vec_embeddings is optional
  }
}

export async function upsertLibrary(
  db: any,
  filePath: string,
  title: string,
  category: string,
  fileType: string,
  text: string,
  hash: string,
  metadata: Record<string, unknown>,
  model: string,
  lang?: string | null,
  folderPath?: string | null,
): Promise<LibraryUpsertResult> {
  const existing = (await db.select(`SELECT id, hash, chunker_version FROM library WHERE file_path = $1`, [
    filePath,
  ])) as Array<{ id: number; hash: string; chunker_version?: number }>;

  if (existing.length > 0) {
    const { id: libraryId, hash: storedHash } = existing[0];

    // Проверяем версию чанкера — если не совпадает, нужна переиндексация
    const chunkerVersionMismatch = (existing[0].chunker_version ?? 0) !== CHUNKER_VERSION;

    if (storedHash === hash && !chunkerVersionMismatch) {
      const embCount = (await db.select(
        `SELECT COUNT(*) AS cnt FROM embeddings WHERE library_id = $1 AND model = $2`,
        [libraryId, model],
      )) as Array<{ cnt: number }>;

      if ((embCount?.[0]?.cnt ?? 0) > 0) {
        // Хеш совпал, версия чанкера совпадает и эмбеддинги есть
        if (lang !== undefined || folderPath !== undefined) {
          await db.execute(
            `UPDATE library SET lang = COALESCE($1, lang), folder_path = COALESCE($2, folder_path),
             updated_at = CURRENT_TIMESTAMP WHERE id = $3`,
            [lang ?? null, folderPath ?? null, libraryId],
          );
        }
        return { libraryId, isNew: false, needsReindex: false };
      }

      await cleanupDocumentIndex(db, libraryId);
      return { libraryId, isNew: false, needsReindex: true };
    }

    await db.execute(
      `UPDATE library SET title=$1, content_raw=$2, hash=$3, category=$4,
       metadata=$5, file_type=$6, lang=$7, folder_path=$8, chunker_version=$9,
       indexed_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
       WHERE id=$10`,
      [title, text, hash, category, JSON.stringify(metadata), fileType, lang ?? null, folderPath ?? null, CHUNKER_VERSION, libraryId],
    );

    await cleanupDocumentIndex(db, libraryId);
    return { libraryId, isNew: false, needsReindex: true };
  }

  const result = await db.execute(
    `INSERT INTO library (title, file_path, file_type, category, content_raw, hash, metadata, lang, folder_path, indexed_at, chunker_version)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,CURRENT_TIMESTAMP,$10)`,
    [title, filePath, fileType, category, text, hash, JSON.stringify(metadata), lang ?? null, folderPath ?? null, CHUNKER_VERSION],
  );

  return { libraryId: Number(result.lastInsertId), isNew: true, needsReindex: true };
}
