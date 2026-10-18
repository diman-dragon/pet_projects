import { ConnectionManager } from '../db/ConnectionManager';
import { AuditService } from '../audit/AuditService';
import type { UserProfile } from '../../types/entities';

export interface GrantCaseAccessInput {
  caseId: number;
  userId: number;
  reason: string;
  approvedBy: UserProfile;
  requestedBy?: number | null;
  expiresAt?: string | null;
}

export const CaseAccessService = {
  async hasCaseAccess(caseId: number, userId: number): Promise<boolean> {
    const db = ConnectionManager.get('clients_main');
    const rows = await db.select<{ count: number }[]>(
      `SELECT COUNT(*) AS count
       FROM case_access
       WHERE case_id = $1
         AND user_id = $2
         AND revoked_at IS NULL
         AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)`,
      [caseId, userId],
    );
    return (rows[0]?.count ?? 0) > 0;
  },

  async requireCaseAccess(actor: UserProfile, caseId: number, action = 'case:view'): Promise<void> {
    if (!(await this.hasCaseAccess(caseId, actor.id))) {
      throw new Error('Case access requires an active case_access grant');
    }

    await AuditService.logForUser(actor, {
      action,
      entityType: 'case',
      entityId: caseId,
    });
  },

  async grant(input: GrantCaseAccessInput): Promise<number> {
    if (input.approvedBy.id === input.userId) {
      throw new Error('Four-eyes rule violation: approved_by must differ from user_id');
    }
    if (!input.reason.trim()) {
      throw new Error('Case access reason is required');
    }

    const db = ConnectionManager.get('clients_main');
    await db.execute(
      `INSERT INTO case_access (case_id, user_id, reason, requested_by, approved_by, expires_at)
       VALUES ($1, $2, $3, $4, $5, $6)
       ON CONFLICT(case_id, user_id) DO UPDATE SET
         reason = excluded.reason,
         requested_by = excluded.requested_by,
         approved_by = excluded.approved_by,
         expires_at = excluded.expires_at,
         revoked_at = NULL,
         granted_at = CURRENT_TIMESTAMP`,
      [
        input.caseId,
        input.userId,
        input.reason.trim(),
        input.requestedBy ?? null,
        input.approvedBy.id,
        input.expiresAt ?? null,
      ],
    );

    const [grant] = await db.select<{ id: number }[]>(
      `SELECT id FROM case_access WHERE case_id = $1 AND user_id = $2`,
      [input.caseId, input.userId],
    );
    if (!grant) throw new Error('Case access grant was not created');

    await AuditService.logForUser(input.approvedBy, {
      action: 'case:access-granted',
      entityType: 'case',
      entityId: input.caseId,
      details: JSON.stringify({
        userId: input.userId,
        reason: input.reason,
        expiresAt: input.expiresAt ?? null,
      }),
    });

    return grant.id;
  },

  async revoke(actor: UserProfile, caseId: number, userId: number, reason: string): Promise<void> {
    const db = ConnectionManager.get('clients_main');
    await db.execute(
      `UPDATE case_access
       SET revoked_at = CURRENT_TIMESTAMP
       WHERE case_id = $1 AND user_id = $2 AND revoked_at IS NULL`,
      [caseId, userId],
    );
    await db.execute(
      `DELETE FROM pii_key_grants
       WHERE user_id = $1
         AND granted_via_case_access_id IN (SELECT id FROM case_access WHERE case_id = $2)`,
      [userId, caseId],
    );

    await AuditService.logForUser(actor, {
      action: 'case:access-revoked',
      entityType: 'case',
      entityId: caseId,
      details: JSON.stringify({ userId, reason }),
    });
  },

  async revokeExpired(): Promise<number> {
    const db = ConnectionManager.get('clients_main');
    const expired = await db.select<{ id: number }[]>(
      `SELECT id FROM case_access
       WHERE revoked_at IS NULL AND expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP`,
    );
    if (expired.length === 0) return 0;

    const ids = expired.map((item) => item.id);
    for (const id of ids) {
      await db.execute(`UPDATE case_access SET revoked_at = CURRENT_TIMESTAMP WHERE id = $1`, [id]);
      await db.execute(`DELETE FROM pii_key_grants WHERE granted_via_case_access_id = $1`, [id]);
    }
    return ids.length;
  },
};
