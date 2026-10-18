import { ConnectionManager } from '../../db/ConnectionManager';
import { createLogger } from '../../logger/Logger';

const log = createLogger('KnowledgeLibrary');

export async function updateLibraryDocument(
  id: number,
  data: { title?: string; category?: string; metadata?: Record<string, unknown> },
): Promise<void> {
  const db = ConnectionManager.get('kb_main');
  const fields: string[] = [];
  const values: unknown[] = [];

  if (data.title !== undefined) {
    fields.push(`title = $${values.length + 1}`);
    values.push(data.title);
  }
  if (data.category !== undefined) {
    fields.push(`category = $${values.length + 1}`);
    values.push(data.category);
  }
  if (data.metadata !== undefined) {
    fields.push(`metadata = $${values.length + 1}`);
    values.push(JSON.stringify(data.metadata));
  }
  if (fields.length === 0) {
    return;
  }

  fields.push('updated_at = CURRENT_TIMESTAMP');
  values.push(id);
  await db.execute(`UPDATE library SET ${fields.join(', ')} WHERE id = $${values.length}`, values);
  log.info(`Документ обновлён: id=${id}`);
}

export async function deleteLibraryDocument(id: number): Promise<void> {
  const db = ConnectionManager.get('kb_main');
  await db.execute(`DELETE FROM library WHERE id = $1`, [id]);
  log.info(`Документ удалён: id=${id}`);
}
