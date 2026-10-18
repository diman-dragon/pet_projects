import { APP_ROLES, normalizeUserRole } from './permissions';
import type { UserProfile } from '../../types/entities';

export function isProtectedSuperadmin(user: Pick<UserProfile, 'role'> | null | undefined): boolean {
  return normalizeUserRole(user?.role) === APP_ROLES.SUPERADMIN;
}

export function canMutateUserRecord(
  actor: Pick<UserProfile, 'id' | 'role'> | null | undefined,
  target: Pick<UserProfile, 'id' | 'role'> | null | undefined,
): boolean {
  if (!actor || !target) return false;
  if (actor.id === target.id) return false;
  if (isProtectedSuperadmin(target)) return false;

  const actorRole = normalizeUserRole(actor.role);
  const targetRole = normalizeUserRole(target.role);

  if (actorRole === APP_ROLES.SUPERADMIN) {
    return targetRole !== APP_ROLES.SUPERADMIN;
  }

  if (actorRole !== APP_ROLES.ADMIN) {
    return false;
  }

  return targetRole !== APP_ROLES.SUPERADMIN && targetRole !== APP_ROLES.ADMIN;
}
