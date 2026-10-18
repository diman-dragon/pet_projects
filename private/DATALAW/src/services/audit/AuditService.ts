import { ConnectionManager } from '../db/ConnectionManager';
import type { UserProfile } from '../../types/entities';

export interface AuditEntryInput {
  actorId?: number | null;
  actorLogin?: string | null;
  actorRole?: string | null;
  action: string;
  entityType: string;
  entityId?: number | string | null;
  entityLabel?: string | null;
  details?: string | null;
}

export interface AuditEntryRecord {
  id: number;
  actor_id: number | null;
  actor_login: string | null;
  actor_role: string | null;
  action: string;
  entity_type: string;
  entity_id: string | null;
  entity_label: string | null;
  details: string | null;
  created_at: string;
}

const DB_KEY = 'user_system';
const encoder = new TextEncoder();

function toHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export const AuditService = {
  async log(entry: AuditEntryInput): Promise<void> {
    if (!ConnectionManager.exists(DB_KEY)) return;

    const db = ConnectionManager.get(DB_KEY);
    await db.execute(
      `INSERT INTO audit_log (
        actor_id,
        actor_login,
        actor_role,
        action,
        entity_type,
        entity_id,
        entity_label,
        details,
        created_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)`,
      [
        entry.actorId ?? null,
        entry.actorLogin ?? null,
        entry.actorRole ?? null,
        entry.action,
        entry.entityType,
        entry.entityId == null ? null : String(entry.entityId),
        entry.entityLabel ?? null,
        entry.details ?? null,
      ],
    );
  },

  async logForUser(
    actor: UserProfile | null | undefined,
    entry: Omit<AuditEntryInput, 'actorId' | 'actorLogin' | 'actorRole'>,
  ): Promise<void> {
    await this.log({
      actorId: actor?.id ?? null,
      actorLogin: actor?.login ?? null,
      actorRole: actor?.role ?? null,
      ...entry,
    });
  },

  async listEntries(limit = 100): Promise<AuditEntryRecord[]> {
    if (!ConnectionManager.exists(DB_KEY)) return [];
    const db = ConnectionManager.get(DB_KEY);
    return db.select<AuditEntryRecord[]>(
      `SELECT id, actor_id, actor_login, actor_role, action, entity_type, entity_id, entity_label, details, created_at
       FROM audit_log
       ORDER BY created_at DESC, id DESC
       LIMIT $1`,
      [limit],
    );
  },

  async listAllEntries(): Promise<AuditEntryRecord[]> {
    if (!ConnectionManager.exists(DB_KEY)) return [];
    const db = ConnectionManager.get(DB_KEY);
    return db.select<AuditEntryRecord[]>(
      `SELECT id, actor_id, actor_login, actor_role, action, entity_type, entity_id, entity_label, details, created_at
       FROM audit_log
       ORDER BY created_at DESC, id DESC`,
    );
  },

  async createDailySnapshot(secret: string, snapshotDate = new Date().toISOString().slice(0, 10)): Promise<void> {
    if (!ConnectionManager.exists(DB_KEY)) return;
    const db = ConnectionManager.get(DB_KEY);
    const entries = await db.select<AuditEntryRecord[]>(
      `SELECT id, actor_id, actor_login, actor_role, action, entity_type, entity_id, entity_label, details, created_at
       FROM audit_log
       WHERE date(created_at) <= date($1)
       ORDER BY id ASC`,
      [snapshotDate],
    );

    const key = await crypto.subtle.importKey(
      'raw',
      encoder.encode(secret),
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['sign'],
    );
    const payload = JSON.stringify(entries);
    const signature = await crypto.subtle.sign('HMAC', key, encoder.encode(payload));
    const hmacHex = toHex(new Uint8Array(signature));
    const lastAuditId = entries.length > 0 ? entries[entries.length - 1].id : null;

    await db.execute(
      `INSERT OR REPLACE INTO audit_snapshots (snapshot_date, entries_count, last_audit_id, hmac_hex)
       VALUES ($1, $2, $3, $4)`,
      [snapshotDate, entries.length, lastAuditId, hmacHex],
    );
  },
};
