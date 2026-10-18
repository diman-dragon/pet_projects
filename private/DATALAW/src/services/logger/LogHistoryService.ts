import { ConnectionManager } from '../db/ConnectionManager';

export interface AppLogRecord {
  id: number;
  level: string;
  tag: string;
  message: string;
  details: string | null;
  created_at: string;
}

export interface AppLogQuery {
  limit?: number;
  level?: string;
  search?: string;
  tag?: string;
}

export const LogHistoryService = {
  async getRecentLogs(limit = 200): Promise<AppLogRecord[]> {
    if (!ConnectionManager.exists('user_system')) return [];

    const db = ConnectionManager.get('user_system');
    return db.select<AppLogRecord[]>(
      `SELECT id, level, tag, message, details, created_at
       FROM app_logs
       ORDER BY created_at DESC, id DESC
       LIMIT $1`,
      [limit]
    );
  },

  async getAllLogs(): Promise<AppLogRecord[]> {
    if (!ConnectionManager.exists('user_system')) return [];

    const db = ConnectionManager.get('user_system');
    return db.select<AppLogRecord[]>(
      `SELECT id, level, tag, message, details, created_at
       FROM app_logs
       ORDER BY created_at DESC, id DESC`,
    );
  },

  async queryLogs(query: AppLogQuery = {}): Promise<AppLogRecord[]> {
    if (!ConnectionManager.exists('user_system')) return [];

    const db = ConnectionManager.get('user_system');
    const where: string[] = [];
    const params: Array<string | number> = [];

    if (query.level && query.level !== 'ALL') {
      where.push('level = $' + (params.length + 1));
      params.push(query.level);
    }

    if (query.tag?.trim()) {
      where.push('tag = $' + (params.length + 1));
      params.push(query.tag.trim());
    }

    if (query.search?.trim()) {
      const value = `%${query.search.trim().toLowerCase()}%`;
      where.push(
        `(LOWER(message) LIKE $${params.length + 1} OR LOWER(COALESCE(details, '')) LIKE $${params.length + 1} OR LOWER(tag) LIKE $${params.length + 1})`,
      );
      params.push(value);
    }

    const limit = Math.max(1, Math.min(query.limit ?? 500, 5000));
    const whereSql = where.length > 0 ? `WHERE ${where.join(' AND ')}` : '';

    return db.select<AppLogRecord[]>(
      `SELECT id, level, tag, message, details, created_at
       FROM app_logs
       ${whereSql}
       ORDER BY created_at DESC, id DESC
       LIMIT $${params.length + 1}`,
      [...params, limit],
    );
  },

  async getLogStats(): Promise<{
    total: number;
    critical: number;
    error: number;
    warn: number;
    info: number;
    debug: number;
    tags: string[];
  }> {
    if (!ConnectionManager.exists('user_system')) {
      return { total: 0, critical: 0, error: 0, warn: 0, info: 0, debug: 0, tags: [] };
    }

    const db = ConnectionManager.get('user_system');
    const [totals] = await db.select<Array<{
      total: number;
      critical: number;
      error: number;
      warn: number;
      info: number;
      debug: number;
    }>>(
      `SELECT
         COUNT(*) AS total,
         SUM(CASE WHEN level = 'CRITICAL' THEN 1 ELSE 0 END) AS critical,
         SUM(CASE WHEN level = 'ERROR' THEN 1 ELSE 0 END) AS error,
         SUM(CASE WHEN level = 'WARN' THEN 1 ELSE 0 END) AS warn,
         SUM(CASE WHEN level = 'INFO' THEN 1 ELSE 0 END) AS info,
         SUM(CASE WHEN level = 'DEBUG' THEN 1 ELSE 0 END) AS debug
       FROM app_logs`,
    );

    const tags = await db.select<Array<{ tag: string }>>(
      `SELECT DISTINCT tag
       FROM app_logs
       ORDER BY tag ASC
       LIMIT 200`,
    );

    return {
      total: Number(totals?.total ?? 0),
      critical: Number(totals?.critical ?? 0),
      error: Number(totals?.error ?? 0),
      warn: Number(totals?.warn ?? 0),
      info: Number(totals?.info ?? 0),
      debug: Number(totals?.debug ?? 0),
      tags: tags.map((row) => row.tag),
    };
  },

  async clearLogs(): Promise<void> {
    if (!ConnectionManager.exists('user_system')) return;
    const db = ConnectionManager.get('user_system');
    await db.execute('DELETE FROM app_logs');
  }
};
