// src/services/clientSearch/ClientIndexService.ts
//
// Индексирует клиентов, дела и обращения в Knowledge Base (RAG).
//
// ИТЕРАЦИЯ 2:
//   - Богатый текст виртуальных документов (больше полей, лучше BM25)
//   - Единая точка индексации через DocumentIndexer (lang + folder_path)
//   - category наследуется по иерархии сущностей:
//       clients/{id}      → 'clients'
//       cases/{id}        → по case_type (litigation/contract/consultation/other)
//       appeals/{id}      → 'appeals'
//   - Согласованные metadata: { entityType, entityId, clientId?, caseId? }
//   - lang проставляется как 'ru' для виртуальных документов (контент всегда русский)

import { invoke } from '@tauri-apps/api/core';
import { ConnectionManager } from '../db/ConnectionManager';
import { createLogger } from '../logger/Logger';
import { EmbeddingModelService } from '../ai/EmbeddingModelService';
import { ensureVecEmbeddingsTable } from '../rag/search/vecTable';
import { EmbeddingService } from '../rag/EmbeddingService';

const log = createLogger('ClientIndexService');

export const CLIENT_CATEGORY = 'clients';
export const CASE_CATEGORY = 'cases';
export const APPEAL_CATEGORY = 'appeals';

// ── Типы ──────────────────────────────────────────────────────────────────────

interface ClientForIndex {
  id: number;
  full_name: string;
  email?: string | null;
  phone?: string | null;
  tax_id?: string | null;
  type?: string | null;
  status?: string | null;
  summary?: string | null;
  client_data?: string | null;
}

interface CaseForIndex {
  id: number;
  client_id: number;
  title: string;
  status: string;
  case_type?: string | null;
  counterparty?: string | null;
  litigation_subtype?: string | null;
  contract_date?: string | null;
  contract_type?: string | null;
  summary?: string | null;
}

interface AppealForIndex {
  id: number;
  line_id: number;
  title: string;
  status?: string | null;
  agency_code?: string | null;
  incoming_number?: string | null;
  outgoing_number?: string | null;
  description?: string | null;
  summary?: string | null;
  /** client_id и case_id нужны для навигации */
  clientId: number;
  caseId: number;
}

// ── Построители текста ────────────────────────────────────────────────────────

function buildClientText(c: ClientForIndex): string {
  const parts: string[] = [`Клиент: ${c.full_name}`];
  if (c.type)   parts.push(`Тип: ${c.type}`);
  if (c.status) parts.push(`Статус: ${c.status}`);
  if (c.tax_id) parts.push(`ИНН: ${c.tax_id}`);
  if (c.email)  parts.push(`Email: ${c.email}`);
  if (c.phone)  parts.push(`Телефон: ${c.phone}`);
  if (c.summary) parts.push(`\nСводка:\n${c.summary}`);
  if (c.client_data) {
    try {
      const data = JSON.parse(c.client_data);
      const fields: string[] = [];
      for (const key of [
        'name', 'full_name', 'short_name', 'okved', 'okved_name',
        'address', 'region', 'activity', 'status', 'director',
      ]) {
        const v = data[key];
        if (typeof v === 'string' && v.trim()) fields.push(`${key}: ${v.trim()}`);
      }
      if (fields.length) parts.push(`\nРеестр:\n${fields.join('\n')}`);
    } catch {
      parts.push(`\n${c.client_data}`);
    }
  }
  return parts.join('\n');
}

function buildCaseText(c: CaseForIndex, clientName: string): string {
  const parts: string[] = [
    `Дело: ${c.title}`,
    `Клиент: ${clientName}`,
    `Статус: ${c.status}`,
  ];
  if (c.case_type)          parts.push(`Тип: ${c.case_type}`);
  if (c.litigation_subtype) parts.push(`Подтип: ${c.litigation_subtype}`);
  if (c.counterparty)       parts.push(`Контрагент: ${c.counterparty}`);
  if (c.contract_date)      parts.push(`Дата договора: ${c.contract_date}`);
  if (c.contract_type)      parts.push(`Тип договора: ${c.contract_type}`);
  if (c.summary)            parts.push(`\nСводка:\n${c.summary}`);
  return parts.join('\n');
}

function buildAppealText(a: AppealForIndex): string {
  const parts: string[] = [`Обращение: ${a.title}`];
  if (a.status)          parts.push(`Статус: ${a.status}`);
  if (a.agency_code)     parts.push(`Орган: ${a.agency_code}`);
  if (a.incoming_number) parts.push(`Вх. номер: ${a.incoming_number}`);
  if (a.outgoing_number) parts.push(`Исх. номер: ${a.outgoing_number}`);
  if (a.description)     parts.push(`\nОписание:\n${a.description}`);
  if (a.summary)         parts.push(`\nСводка:\n${a.summary}`);
  return parts.join('\n');
}

/**
 * Определяет category дела по case_type.
 * virtual://cases/{id} наследует категорию из типа дела.
 */
function categoryFromCaseType(caseType: string | null | undefined): string {
  if (!caseType) return CASE_CATEGORY;
  const t = caseType.toLowerCase();
  if (t.includes('litigation') || t.includes('суд')) return 'Litigation';
  if (t.includes('contract') || t.includes('договор')) return 'Contracts';
  if (t.includes('consult') || t.includes('консульт')) return 'Consultation';
  return CASE_CATEGORY;
}

// ── Ядро upsert ───────────────────────────────────────────────────────────────

async function simpleHash(text: string): Promise<string> {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, '0')).join('');
}

async function upsertVirtualDoc(
  virtualPath: string,
  title: string,
  category: string,
  text: string,
  meta: Record<string, unknown>,
  /** Язык контента виртуального документа (почти всегда 'ru') */
  lang = 'ru',
): Promise<void> {
  const db = ConnectionManager.get('kb_main');
  const hash = await simpleHash(text);

  const existing = await db.select<{ id: number; hash: string | null }[]>(
    `SELECT id, hash FROM library WHERE file_path = $1 LIMIT 1`,
    [virtualPath],
  );

  let libraryId: number;

  if (existing.length > 0) {
    libraryId = existing[0].id;
    if (existing[0].hash === hash) return; // не изменилось
    await db.execute(
      `UPDATE library SET title=$1, hash=$2, metadata=$3, lang=$4,
       updated_at=CURRENT_TIMESTAMP WHERE id=$5`,
      [title, hash, JSON.stringify(meta), lang, libraryId],
    );
    await db.execute(`DELETE FROM embeddings WHERE library_id=$1`, [libraryId]);
    await db.execute(`DELETE FROM chunks WHERE library_id=$1`, [libraryId]);
  } else {
    const res = await db.execute(
      `INSERT INTO library
         (title, file_path, file_type, category, hash, metadata, lang,
          created_at, updated_at)
       VALUES ($1, $2, 'virtual', $3, $4, $5, $6,
               CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)`,
      [title, virtualPath, category, hash, JSON.stringify(meta), lang],
    );
    libraryId = Number(res.lastInsertId);
  }

  // Чанки + эмбеддинги
  const chunks = await invoke<{
    text: string; chunk_index: number; char_start: number; char_end: number;
  }[]>('rag_chunk_text', {
    text,
    chunk_size: 3200,
    overlap_size: 400,
    min_chunk_size: 600,
    max_chunk_chars: 4200,
  });

  if (!chunks.length) return;

  const model = await EmbeddingModelService.get();

  for (const chunk of chunks) {
    const chunkRes = await db.execute(
      `INSERT INTO chunks (library_id, chunk_index, text, char_start, char_end)
       VALUES ($1,$2,$3,$4,$5)`,
      [libraryId, chunk.chunk_index, chunk.text, chunk.char_start, chunk.char_end],
    );
    const rawId = chunkRes.lastInsertId;
    const chunkId = typeof rawId === 'bigint' ? Number(rawId) : Math.trunc(Number(rawId));
    if (!Number.isInteger(chunkId) || chunkId < 1 || chunkId > 2147483647) {
      throw new Error(`Invalid chunkId for vec0: ${rawId} (got ${chunkId})`);
    }

    const emb = await EmbeddingService.embed(chunk.text, model, false);
    const vecJson = JSON.stringify(Array.from(emb.vector));

    await db.execute(
      `INSERT INTO embeddings (chunk_id, library_id, vector_blob, model, dim)
       VALUES ($1,$2,$3,$4,$5)`,
      [chunkId, libraryId, vecJson, model, emb.dim],
    );
    try {
      await ensureVecEmbeddingsTable(db, emb.dim);
      await db.execute(
        `INSERT INTO vec_embeddings (chunk_id, library_id, model, embedding)
         VALUES (CAST($1 AS INTEGER),CAST($2 AS INTEGER),$3,$4)`,
        [chunkId, libraryId, model, vecJson],
      );
    } catch { /* vec0 недоступна */ }
  }

  log.info(`Virtual indexed: ${title} (${chunks.length} chunks, cat=${category})`);
}

async function removeVirtualDoc(virtualPath: string): Promise<void> {
  const db = ConnectionManager.get('kb_main');
  const rows = await db.select<{ id: number }[]>(
    `SELECT id FROM library WHERE file_path = $1 LIMIT 1`,
    [virtualPath],
  );
  if (rows.length === 0) return;
  await db.execute(`DELETE FROM library WHERE id = $1`, [rows[0].id]);
}

// ── Публичный API ─────────────────────────────────────────────────────────────

export const ClientIndexService = {
  async indexClient(client: ClientForIndex): Promise<void> {
    try {
      await upsertVirtualDoc(
        `virtual://clients/${client.id}`,
        client.full_name,
        CLIENT_CATEGORY,
        buildClientText(client),
        {
          entityType: 'client',
          entityId: client.id,
          type: client.type,
          status: client.status,
          tax_id: client.tax_id,
        },
      );
    } catch (e) {
      log.warn(`Ошибка индексации клиента ${client.id}:`, e);
    }
  },

  async indexCase(c: CaseForIndex, clientName: string): Promise<void> {
    try {
      const category = categoryFromCaseType(c.case_type);
      await upsertVirtualDoc(
        `virtual://cases/${c.id}`,
        `${clientName} — ${c.title}`,
        category,
        buildCaseText(c, clientName),
        {
          entityType: 'case',
          entityId: c.id,
          clientId: c.client_id,
          status: c.status,
          case_type: c.case_type,
        },
      );
    } catch (e) {
      log.warn(`Ошибка индексации дела ${c.id}:`, e);
    }
  },

  async indexAppeal(a: AppealForIndex): Promise<void> {
    try {
      await upsertVirtualDoc(
        `virtual://appeals/${a.id}`,
        a.title,
        APPEAL_CATEGORY,
        buildAppealText(a),
        {
          entityType: 'appeal',
          entityId: a.id,
          clientId: a.clientId,
          caseId: a.caseId,
          lineId: a.line_id,
          status: a.status,
        },
      );
    } catch (e) {
      log.warn(`Ошибка индексации обращения ${a.id}:`, e);
    }
  },

  async indexAllClients(
    clients: ClientForIndex[],
    onProgress?: (done: number, total: number, name: string) => void,
  ): Promise<{ done: number; errors: string[] }> {
    const errors: string[] = [];
    let done = 0;
    for (const client of clients) {
      try {
        onProgress?.(done, clients.length, client.full_name);
        await ClientIndexService.indexClient(client);
        done++;
      } catch (e: any) {
        errors.push(`${client.full_name}: ${e?.message ?? String(e)}`);
      }
    }
    return { done, errors };
  },

  async removeClient(clientId: number): Promise<void> {
    await removeVirtualDoc(`virtual://clients/${clientId}`);
  },

  async removeCase(caseId: number): Promise<void> {
    await removeVirtualDoc(`virtual://cases/${caseId}`);
  },

  async removeAppeal(appealId: number): Promise<void> {
    await removeVirtualDoc(`virtual://appeals/${appealId}`);
  },
};
