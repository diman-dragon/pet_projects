import { ConnectionManager } from '../db/ConnectionManager';
import { legalChunkText } from '../rag/native/legalChunkText';
import type { LegalActInput } from './types';

export const LegalCorpusIngestService = {
  async ingestAct(input: LegalActInput): Promise<number> {
    const db = ConnectionManager.get('legal_corpus');

    await db.execute(
      `INSERT OR IGNORE INTO jurisdictions (code, name, legal_system)
       VALUES ($1, $2, $3)`,
      [input.jurisdictionCode, input.jurisdictionName, input.legalSystem ?? null],
    );
    const [jurisdiction] = await db.select<{ id: number }[]>(
      `SELECT id FROM jurisdictions WHERE code = $1`,
      [input.jurisdictionCode],
    );
    if (!jurisdiction) throw new Error('Jurisdiction was not created');

    let courtBodyId: number | null = null;
    if (input.courtBodyName) {
      await db.execute(
        `INSERT INTO courts_bodies (jurisdiction_id, name, level, code)
         SELECT $1, $2, $3, $4
         WHERE NOT EXISTS (
           SELECT 1 FROM courts_bodies WHERE jurisdiction_id = $1 AND name = $2
         )`,
        [jurisdiction.id, input.courtBodyName, input.courtBodyLevel ?? null, input.courtBodyCode ?? null],
      );
      const [court] = await db.select<{ id: number }[]>(
        `SELECT id FROM courts_bodies WHERE jurisdiction_id = $1 AND name = $2`,
        [jurisdiction.id, input.courtBodyName],
      );
      courtBodyId = court?.id ?? null;
    }

    await db.execute(
      `INSERT INTO acts (
        jurisdiction_id, court_body_id, act_type, title, act_number, adopted_date,
        effective_from, effective_to, status, lang, source_url, content_raw, hash
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)`,
      [
        jurisdiction.id,
        courtBodyId,
        input.actType,
        input.title,
        input.actNumber ?? null,
        input.adoptedDate ?? null,
        input.effectiveFrom ?? null,
        input.effectiveTo ?? null,
        input.status ?? 'IN_FORCE',
        input.lang ?? null,
        input.sourceUrl ?? null,
        input.contentRaw,
        input.hash,
      ],
    );

    const [act] = await db.select<{ id: number }[]>(
      `SELECT id FROM acts WHERE hash = $1 ORDER BY id DESC LIMIT 1`,
      [input.hash],
    );
    if (!act) throw new Error('Act was not created');

    await db.execute(
      `INSERT INTO act_versions (act_id, version_label, valid_from, valid_to, content_raw)
       VALUES ($1, $2, $3, $4, $5)`,
      [act.id, input.effectiveFrom ?? 'unknown', input.effectiveFrom ?? '0000-01-01', input.effectiveTo ?? null, input.contentRaw],
    );
    const [version] = await db.select<{ id: number }[]>(
      `SELECT id FROM act_versions WHERE act_id = $1 ORDER BY id DESC LIMIT 1`,
      [act.id],
    );

    const chunkResult = await legalChunkText(input.contentRaw, { docTitle: input.title });
    const chunks = chunkResult.chunks.length > 0
      ? chunkResult.chunks
      : [{ text: input.contentRaw, chunkIndex: 0, charStart: 0, charEnd: input.contentRaw.length }];

    for (const [index, chunk] of chunks.entries()) {
      await db.execute(
        `INSERT INTO act_chunks (act_id, version_id, chunk_index, text, article_ref)
         VALUES ($1, $2, $3, $4, $5)`,
        [act.id, version?.id ?? null, chunk.chunkIndex ?? index, chunk.text, chunk.articleRef ?? null],
      );
    }

    return act.id;
  },
};
