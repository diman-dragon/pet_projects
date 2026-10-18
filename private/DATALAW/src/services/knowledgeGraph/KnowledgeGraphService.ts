// src/services/knowledgeGraph/KnowledgeGraphService.ts
//
// Сервис Knowledge Graph — оркестратор извлечения, хранения и запроса
// сущностей и отношений из документов.
//
// Архитектура:
//   KnowledgeGraphAgent  → извлекает KgEntity[] + KgRelation[] из текста
//   KnowledgeGraphService → сохраняет в SQLite таблицы kg_*
//
// Интеграция с индексером:
//   При вызове indexSingleFile/chunkWriter уже создан chunk.
//   KnowledgeGraphService.extractFromChunk() вызывается ПОСЛЕ записи чанка.
//   Это обеспечивает chunk_id для kg_entity_sources.

import { ConnectionManager } from '../db/ConnectionManager';
import { KnowledgeGraphAgent } from '../ai/agents/KnowledgeGraphAgent';
import type {
  KgEntity,
  KgRelation,
  KgExtractionResult,
} from '../ai/agents/KnowledgeGraphAgent';
import { createLogger } from '../logger/Logger';

const log = createLogger('KnowledgeGraphService');

function actorFilterSql(alias = 'e'): string {
  return `${alias}.entity_type IN ('PERSON', 'ORGANIZATION')`;
}

// ─── Типы для запросов к БД ───────────────────────────────────────────────────

export interface KgEntityRow {
  id: number;
  name: string;
  entity_type: string;
  canonical_name: string;
  description: string | null;
  confidence: number;
  created_at: string;
  updated_at: string;
}

export interface KgEntityTableRow extends KgEntityRow {
  relation_count: number;
  source_count: number;
  document_count: number;
  chunk_count: number;
}

export interface KgRelationRow {
  id: number;
  source_id: number;
  target_id: number;
  relation_type: string;
  label: string | null;
  weight: number;
  confidence: number;
  created_at: string;
  /** Дополнительно — из JOIN с kg_entities */
  source_name?: string;
  target_name?: string;
}

export interface KgGraphData {
  entities: KgEntityRow[];
  relations: KgRelationRow[];
}

export interface KgExtractionStats {
  entitiesFound: number;
  relationsFound: number;
  usedLlm: boolean;
  durationMs: number;
}

// ─── Сервис ───────────────────────────────────────────────────────────────────

export const KnowledgeGraphService = {

  // ── Внутренние хелперы ─────────────────────────────────────────────────────

  _getDb() {
    return ConnectionManager.get('kb_main');
  },

  /**
   * Вставляет или возвращает существующую сущность по canonical_name.
   * Возвращает id записи в kg_entities.
   */
  async _upsertEntity(entity: KgEntity): Promise<number> {
    const db = this._getDb();

    // Проверяем существование
    const existing = await db.select<{ id: number }[]>(
      `SELECT id FROM kg_entities WHERE canonical_name = ? LIMIT 1`,
      [entity.canonical_name],
    );

    if (existing.length > 0) {
      // Keep the most descriptive version while preserving the strongest confidence.
      await db.execute(
        `UPDATE kg_entities
         SET name = CASE
               WHEN LENGTH(TRIM(COALESCE(name, ''))) >= LENGTH(TRIM(?)) THEN name
               ELSE ?
             END,
             entity_type = CASE
               WHEN entity_type = 'OTHER' AND ? != 'OTHER' THEN ?
               ELSE entity_type
             END,
             description = CASE
               WHEN COALESCE(LENGTH(TRIM(description)), 0) >= COALESCE(LENGTH(TRIM(?)), 0) THEN description
               ELSE ?
             END,
             confidence = MAX(confidence, ?),
             updated_at = CURRENT_TIMESTAMP
         WHERE id = ?`,
        [
          entity.name,
          entity.name,
          entity.entity_type,
          entity.entity_type,
          entity.description ?? null,
          entity.description ?? null,
          entity.confidence,
          existing[0].id,
        ],
      );
      return existing[0].id;
    }

    // Вставляем новую
    const result = await db.execute(
      `INSERT INTO kg_entities (name, entity_type, canonical_name, description, confidence)
       VALUES (?, ?, ?, ?, ?)`,
      [
        entity.name,
        entity.entity_type,
        entity.canonical_name,
        entity.description ?? null,
        entity.confidence,
      ],
    );
    return result.lastInsertId as number;
  },

  /**
   * Вставляет отношение (INSERT OR IGNORE по unique index на source+target+type).
   */
  async _upsertRelation(
    sourceId: number,
    targetId: number,
    relation: KgRelation,
  ): Promise<void> {
    const db = this._getDb();
    await db.execute(
      `INSERT INTO kg_relations
         (source_id, target_id, relation_type, label, weight, confidence)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
         label = COALESCE(kg_relations.label, excluded.label),
         weight = MAX(kg_relations.weight, excluded.weight),
         confidence = MAX(kg_relations.confidence, excluded.confidence)`,
      [
        sourceId,
        targetId,
        relation.relation_type,
        relation.label ?? null,
        relation.weight,
        relation.confidence,
      ],
    );
  },

  /**
   * Привязывает сущность к источнику (library/chunk/client/case).
   */
  async _linkSource(
    entityId: number,
    opts: {
      libraryId?: number;
      chunkId?: number;
      clientId?: number;
      caseId?: number;
      contextSnippet?: string;
    },
  ): Promise<void> {
    const db = this._getDb();
    await db.execute(
      `INSERT OR IGNORE INTO kg_entity_sources
         (entity_id, library_id, chunk_id, client_id, case_id, context_snippet)
       VALUES (?, ?, ?, ?, ?, ?)`,
      [
        entityId,
        opts.libraryId ?? null,
        opts.chunkId ?? null,
        opts.clientId ?? null,
        opts.caseId ?? null,
        opts.contextSnippet ?? null,
      ],
    );
  },

  // ── Публичные методы ───────────────────────────────────────────────────────

  /**
   * Основной метод интеграции с индексером.
   * Извлекает сущности и отношения из текста чанка,
   * сохраняет в kg_* таблицы, записывает в журнал.
   *
   * @param text      Текст чанка
   * @param chunkId   ID записи в таблице chunks (может быть null для виртуальных)
   * @param libraryId ID документа в таблице library
   * @param opts      Дополнительный контекст (clientId, caseId)
   */
  async extractFromChunk(
    text: string,
    chunkId: number | null,
    libraryId: number,
    opts: { clientId?: number; caseId?: number } = {},
  ): Promise<KgExtractionStats> {
    const db = this._getDb();
    const started = Date.now();

    // Создаём запись в журнале заданий
    const jobResult = await db.execute(
      `INSERT INTO kg_extraction_jobs (library_id, chunk_id, status)
       VALUES (?, ?, 'running')`,
      [libraryId, chunkId ?? null],
    );
    const jobId = jobResult.lastInsertId as number;

    let extraction: KgExtractionResult = { entities: [], relations: [], usedLlm: false };

    try {
      extraction = await KnowledgeGraphAgent.extract(text);

      // Шаг 1: Вставляем сущности и строим карту canonical → id
      const entityIdMap = new Map<string, number>();
      for (const entity of extraction.entities) {
        const id = await this._upsertEntity(entity);
        entityIdMap.set(entity.canonical_name, id);

        // Привязываем к источнику
        const snippet = text.slice(0, 200).replace(/\n/g, ' ').trim();
        await this._linkSource(id, {
          libraryId,
          chunkId: chunkId ?? undefined,
          clientId: opts.clientId,
          caseId: opts.caseId,
          contextSnippet: snippet,
        });
      }

      // Шаг 2: Вставляем отношения
      for (const rel of extraction.relations) {
        const sourceId = entityIdMap.get(rel.source);
        const targetId = entityIdMap.get(rel.target);
        if (sourceId !== undefined && targetId !== undefined) {
          await this._upsertRelation(sourceId, targetId, rel);
        }
      }

      const durationMs = Date.now() - started;

      // Обновляем журнал — успех
      await db.execute(
        `UPDATE kg_extraction_jobs
         SET status = 'done',
             entities_found = ?,
             relations_found = ?,
             model_used = ?,
             duration_ms = ?,
             finished_at = CURRENT_TIMESTAMP
         WHERE id = ?`,
        [
          extraction.entities.length,
          extraction.relations.length,
          extraction.usedLlm ? 'llm' : 'deterministic',
          durationMs,
          jobId,
        ],
      );

      return {
        entitiesFound: extraction.entities.length,
        relationsFound: extraction.relations.length,
        usedLlm: extraction.usedLlm,
        durationMs,
      };
    } catch (err) {
      log.error('KnowledgeGraphService.extractFromChunk failed', err);

      await db.execute(
        `UPDATE kg_extraction_jobs
         SET status = 'failed',
             error_message = ?,
             finished_at = CURRENT_TIMESTAMP
         WHERE id = ?`,
        [String(err), jobId],
      );

      return {
        entitiesFound: 0,
        relationsFound: 0,
        usedLlm: false,
        durationMs: Date.now() - started,
      };
    }
  },

  /**
   * Возвращает граф (сущности + отношения) для указанного документа.
   *
   * @param libraryId  ID документа в library
   */
  async getGraphForDocument(libraryId: number): Promise<KgGraphData> {
    const db = this._getDb();

    const entities = await db.select<KgEntityRow[]>(
      `SELECT DISTINCT e.*
       FROM kg_entities e
       JOIN kg_entity_sources s ON s.entity_id = e.id
       WHERE s.library_id = ?
         AND ${actorFilterSql('e')}
       ORDER BY e.entity_type, e.name`,
      [libraryId],
    );

    if (entities.length === 0) {
      return { entities: [], relations: [] };
    }

    const ids = entities.map((e) => e.id);
    const placeholders = ids.map(() => '?').join(',');

    const relations = await db.select<KgRelationRow[]>(
      `SELECT r.*,
              s.name AS source_name,
              t.name AS target_name
       FROM kg_relations r
       JOIN kg_entities s ON s.id = r.source_id
       JOIN kg_entities t ON t.id = r.target_id
       WHERE r.source_id IN (${placeholders})
         AND r.target_id IN (${placeholders})
       ORDER BY r.weight DESC`,
      [...ids, ...ids],
    );

    return { entities, relations };
  },

  /**
   * Полнотекстовый поиск сущностей по имени.
   *
   * @param query      Поисковый запрос
   * @param entityType Фильтр по типу (опционально)
   * @param limit      Максимум результатов (default: 20)
   */
  async searchEntities(
    query: string,
    entityType?: string,
    limit = 20,
  ): Promise<KgEntityRow[]> {
    const db = this._getDb();
    const ftsQuery = query
      .trim()
      .split(/\s+/)
      .map((w) => `"${w}"*`)
      .join(' OR ');

    if (entityType) {
      return db.select<KgEntityRow[]>(
      `SELECT e.*
       FROM kg_entities e
       JOIN kg_entities_fts fts ON fts.rowid = e.id
       WHERE kg_entities_fts MATCH ?
         AND ${actorFilterSql('e')}
           AND e.entity_type = ?
         ORDER BY rank
         LIMIT ?`,
        [ftsQuery, entityType, limit],
      );
    }

    return db.select<KgEntityRow[]>(
      `SELECT e.*
       FROM kg_entities e
       JOIN kg_entities_fts fts ON fts.rowid = e.id
       WHERE kg_entities_fts MATCH ?
         AND ${actorFilterSql('e')}
       ORDER BY rank
       LIMIT ?`,
      [ftsQuery, limit],
    );
  },

  /**
   * Возвращает N-hop окружение (соседей) вокруг сущности.
   *
   * @param entityId  ID центральной сущности
   * @param hops      Глубина (1 или 2, default: 1)
   */
  async getNeighbors(
    entityId: number,
    hops: 1 | 2 = 1,
  ): Promise<KgGraphData> {
    const db = this._getDb();

    // 1-hop: все сущности напрямую связанные с entityId
    const neighbors1 = await db.select<KgEntityRow[]>(
      `SELECT DISTINCT e.*
       FROM kg_entities e
       JOIN kg_relations r ON (r.source_id = e.id OR r.target_id = e.id)
       WHERE (r.source_id = ? OR r.target_id = ?)
         AND ${actorFilterSql('e')}
         AND e.id != ?`,
      [entityId, entityId, entityId],
    );

    let allEntities = neighbors1;

    if (hops === 2 && neighbors1.length > 0) {
      const ids1 = neighbors1.map((e) => e.id);
      const ph = ids1.map(() => '?').join(',');
      const neighbors2 = await db.select<KgEntityRow[]>(
        `SELECT DISTINCT e.*
         FROM kg_entities e
         JOIN kg_relations r ON (r.source_id = e.id OR r.target_id = e.id)
         WHERE (r.source_id IN (${ph}) OR r.target_id IN (${ph}))
           AND ${actorFilterSql('e')}
           AND e.id NOT IN (${ph})
           AND e.id != ?`,
        [...ids1, ...ids1, ...ids1, entityId],
      );
      allEntities = [...neighbors1, ...neighbors2];
    }

    // Включаем саму центральную сущность
    const center = await db.select<KgEntityRow[]>(
      `SELECT * FROM kg_entities WHERE id = ? AND ${actorFilterSql('kg_entities')}`,
      [entityId],
    );
    allEntities = [...center, ...allEntities];

    const ids = allEntities.map((e) => e.id);
    if (ids.length === 0) return { entities: [], relations: [] };

    const ph = ids.map(() => '?').join(',');
    const relations = await db.select<KgRelationRow[]>(
      `SELECT r.*,
              s.name AS source_name,
              t.name AS target_name
       FROM kg_relations r
       JOIN kg_entities s ON s.id = r.source_id
       JOIN kg_entities t ON t.id = r.target_id
       WHERE r.source_id IN (${ph})
         AND r.target_id IN (${ph})`,
      [...ids, ...ids],
    );

    return { entities: allEntities, relations };
  },

  /**
   * Статистика графа (для дашборда).
   */
  async getStats(): Promise<{
    totalEntities: number;
    totalRelations: number;
    byType: Record<string, number>;
    pendingJobs: number;
    completedJobs: number;
    failedJobs: number;
    totalJobs: number;
    progressPercent: number;
  }> {
    const db = this._getDb();

    const [totals] = await db.select<{ e: number; r: number }[]>(
      `SELECT
         (SELECT COUNT(*) FROM kg_entities e WHERE ${actorFilterSql('e')}) AS e,
         (
           SELECT COUNT(*)
           FROM kg_relations r
           JOIN kg_entities s ON s.id = r.source_id
           JOIN kg_entities t ON t.id = r.target_id
           WHERE ${actorFilterSql('s')} AND ${actorFilterSql('t')}
         ) AS r`,
      [],
    );

    const byTypeRows = await db.select<{ entity_type: string; cnt: number }[]>(
      `SELECT entity_type, COUNT(*) AS cnt
       FROM kg_entities
       WHERE ${actorFilterSql('kg_entities')}
       GROUP BY entity_type
       ORDER BY cnt DESC`,
      [],
    );

    const [jobRow] = await db.select<{ pending: number; completed: number; failed: number; total: number }[]>(
      `SELECT
         SUM(CASE WHEN status IN ('pending', 'running') THEN 1 ELSE 0 END) AS pending,
         SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) AS completed,
         SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
         COUNT(*) AS total
       FROM kg_extraction_jobs`,
      [],
    );

    const totalJobs = jobRow?.total ?? 0;
    const completedJobs = jobRow?.completed ?? 0;
    const progressPercent = totalJobs > 0 ? Math.round((completedJobs / totalJobs) * 100) : 0;

    return {
      totalEntities: totals?.e ?? 0,
      totalRelations: totals?.r ?? 0,
      byType: Object.fromEntries(byTypeRows.map((r) => [r.entity_type, r.cnt])),
      pendingJobs: jobRow?.pending ?? 0,
      completedJobs,
      failedJobs: jobRow?.failed ?? 0,
      totalJobs,
      progressPercent,
    };
  },

  /**
   * Возвращает все сущности (с опциональным фильтром по типу).
   * Используется в KnowledgeGraphPage когда поиск пустой.
   */
  async getAllEntities(entityType?: string, limit = 300): Promise<KgEntityRow[]> {
    const db = this._getDb();
    if (entityType) {
      return db.select<KgEntityRow[]>(
        `SELECT *
         FROM kg_entities
         WHERE ${actorFilterSql('kg_entities')} AND entity_type = ?
         ORDER BY confidence DESC, name ASC
         LIMIT ?`,
        [entityType, limit],
      );
    }
    return db.select<KgEntityRow[]>(
      `SELECT *
       FROM kg_entities
       WHERE ${actorFilterSql('kg_entities')}
       ORDER BY confidence DESC, name ASC
       LIMIT ?`,
      [limit],
    );
  },

  async getEntityTable(
    entityType?: string,
    search?: string,
    limit = 300,
  ): Promise<KgEntityTableRow[]> {
    const db = this._getDb();
    const hasSearch = Boolean(search?.trim());
    const trimmedSearch = search?.trim() ?? '';
    const ftsQuery = trimmedSearch
      .split(/\s+/)
      .filter(Boolean)
      .map((word) => `"${word}"*`)
      .join(' OR ');

    const where: string[] = [];
    const params: Array<string | number> = [];

    if (hasSearch) {
      where.push(`kg_entities_fts MATCH ?`);
      params.push(ftsQuery);
    }
    if (entityType) {
      where.push(`e.entity_type = ?`);
      params.push(entityType);
    }
    where.push(actorFilterSql('e'));

    const joinFts = hasSearch
      ? `JOIN kg_entities_fts ON kg_entities_fts.rowid = e.id`
      : '';
    const whereSql = where.length > 0 ? `WHERE ${where.join(' AND ')}` : '';

    return db.select<KgEntityTableRow[]>(
      `SELECT
         e.*,
         (
           SELECT COUNT(*)
           FROM kg_relations r
           WHERE r.source_id = e.id OR r.target_id = e.id
         ) AS relation_count,
         (
           SELECT COUNT(*)
           FROM kg_entity_sources s
           WHERE s.entity_id = e.id
         ) AS source_count,
         (
           SELECT COUNT(DISTINCT s.library_id)
           FROM kg_entity_sources s
           WHERE s.entity_id = e.id AND s.library_id IS NOT NULL
         ) AS document_count,
         (
           SELECT COUNT(DISTINCT s.chunk_id)
           FROM kg_entity_sources s
           WHERE s.entity_id = e.id AND s.chunk_id IS NOT NULL
         ) AS chunk_count
       FROM kg_entities e
       ${joinFts}
       ${whereSql}
       ORDER BY relation_count DESC, document_count DESC, confidence DESC, name ASC
       LIMIT ?`,
      [...params, limit],
    );
  },

  /**
   * Возвращает связи между указанными сущностями (по массиву id).
   * Безопасная замена прямому вызову _getDb() из UI.
   */
  async getRelationsForEntities(entityIds: number[], limit = 500): Promise<KgRelationRow[]> {
    if (entityIds.length === 0) return [];
    const db = this._getDb();
    const ph = entityIds.map(() => '?').join(',');
    return db.select<KgRelationRow[]>(
      `SELECT r.*, s.name AS source_name, t.name AS target_name
       FROM kg_relations r
       JOIN kg_entities s ON s.id = r.source_id
       JOIN kg_entities t ON t.id = r.target_id
       WHERE r.source_id IN (${ph}) AND r.target_id IN (${ph})
         AND ${actorFilterSql('s')}
         AND ${actorFilterSql('t')}
       ORDER BY r.weight DESC
       LIMIT ?`,
      [...entityIds, ...entityIds, limit],
    );
  },

  /**
   * Полная очистка графа (используется при переиндексации).
   */
  async clearAll(): Promise<void> {
    const db = this._getDb();
    await db.execute(`DELETE FROM kg_relations`);
    await db.execute(`DELETE FROM kg_entity_sources`);
    await db.execute(`DELETE FROM kg_entities`);
    await db.execute(`DELETE FROM kg_extraction_jobs`);
    log.info('KnowledgeGraphService: all data cleared');
  },
};
