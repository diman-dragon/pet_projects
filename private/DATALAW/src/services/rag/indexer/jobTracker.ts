// indexer/jobTracker.ts — управление статусами задач индексации
//
// Шаг 4.2 — статусы: pending | running | done | failed

export type JobStatus = 'pending' | 'running' | 'done' | 'failed';

export async function createJob(db: any, libraryId: number, totalChunks: number): Promise<number> {
  const res = await db.execute(
    `INSERT INTO index_jobs (library_id, status, total_chunks, started_at)
     VALUES ($1,'running',$2,CURRENT_TIMESTAMP)`,
    [libraryId, totalChunks],
  );
  return Number(res.lastInsertId);
}

export async function updateJobProgress(db: any, jobId: number, done: number): Promise<void> {
  await db.execute(`UPDATE index_jobs SET done_chunks=$1 WHERE id=$2`, [done, jobId]);
}

export async function finishJob(db: any, jobId: number): Promise<void> {
  await db.execute(
    `UPDATE index_jobs SET status='done', finished_at=CURRENT_TIMESTAMP WHERE id=$1`,
    [jobId],
  );
}

export async function failJob(db: any, jobId: number, msg: string): Promise<void> {
  await db.execute(
    `UPDATE index_jobs SET status='failed', error_msg=$1, finished_at=CURRENT_TIMESTAMP WHERE id=$2`,
    [msg, jobId],
  );
}
