import type { UserProfile, UserRole } from '../../types/entities';

export type AppPermission =
  | 'dashboard:view'
  | 'clients:view'
  | 'clients:create'
  | 'clients:update'
  | 'clients:delete'
  | 'cases:view'
  | 'cases:create'
  | 'cases:update'
  | 'cases:delete'
  | 'lines:view'
  | 'lines:create'
  | 'lines:update'
  | 'lines:delete'
  | 'appeals:view'
  | 'appeals:create'
  | 'appeals:update'
  | 'appeals:delete'
  | 'knowledge:view'
  | 'knowledge:edit'
  | 'ai:use'
  | 'users:view'
  | 'users:create'
  | 'users:update'
  | 'users:delete'
  | 'users:manage-admins'
  | 'settings:view'
  | 'audit:view'
  | 'trash:purge';

export const APP_ROLES = {
  SUPERADMIN: 'SUPERADMIN',
  ADMIN: 'ADMIN',
  LAWYER: 'LAWYER',
  ASSISTANT: 'ASSISTANT',
  ANALYST: 'ANALYST',
} as const satisfies Record<string, UserRole>;

export const ROLE_LABELS: Record<UserRole, string> = {
  SUPERADMIN: 'Суперадмин',
  ADMIN: 'Админ',
  LAWYER: 'Юрист',
  ASSISTANT: 'Администратор',
  ANALYST: 'Аналитик',
};

const ALL_CASE_PERMISSIONS: AppPermission[] = [
  'dashboard:view',
  'clients:view',
  'clients:create',
  'clients:update',
  'clients:delete',
  'cases:view',
  'cases:create',
  'cases:update',
  'cases:delete',
  'lines:view',
  'lines:create',
  'lines:update',
  'lines:delete',
  'appeals:view',
  'appeals:create',
  'appeals:update',
  'appeals:delete',
];

const PERMISSIONS_BY_ROLE: Record<UserRole, readonly AppPermission[]> = {
  SUPERADMIN: [
    ...ALL_CASE_PERMISSIONS,
    'knowledge:view',
    'knowledge:edit',
    'ai:use',
    'users:view',
    'users:create',
    'users:update',
    'users:delete',
    'users:manage-admins',
    'settings:view',
    'audit:view',
    'trash:purge',
  ],
  ADMIN: [
    ...ALL_CASE_PERMISSIONS,
    'knowledge:view',
    'knowledge:edit',
    'ai:use',
    'users:view',
    'users:create',
    'users:update',
    'users:delete',
    'settings:view',
    'audit:view',
    'trash:purge',
  ],
  LAWYER: [...ALL_CASE_PERMISSIONS, 'knowledge:view', 'knowledge:edit', 'ai:use'],
  ASSISTANT: ['dashboard:view', 'clients:view', 'clients:create', 'clients:update', 'cases:view'],
  ANALYST: ['dashboard:view', 'knowledge:view', 'audit:view'],
};

export function normalizeUserRole(role?: string | null): UserRole {
  const value = role?.toUpperCase().trim();
  if (value === APP_ROLES.SUPERADMIN) return APP_ROLES.SUPERADMIN;
  if (value === APP_ROLES.ADMIN) return APP_ROLES.ADMIN;
  if (value === APP_ROLES.LAWYER) return APP_ROLES.LAWYER;
  if (value === APP_ROLES.ANALYST) return APP_ROLES.ANALYST;
  return APP_ROLES.ASSISTANT;
}

export function getRolePermissions(role?: string | null): readonly AppPermission[] {
  return PERMISSIONS_BY_ROLE[normalizeUserRole(role)];
}

export function hasPermission(
  user: Pick<UserProfile, 'role'> | null | undefined,
  permission: AppPermission,
): boolean {
  if (!user) return false;
  return getRolePermissions(user.role).includes(permission);
}

export function canManageTargetUser(
  actor: Pick<UserProfile, 'role' | 'id'> | null | undefined,
  target: Pick<UserProfile, 'role' | 'id'>,
): boolean {
  if (!actor) return false;
  if (actor.id === target.id) return false;

  const actorRole = normalizeUserRole(actor.role);
  const targetRole = normalizeUserRole(target.role);

  if (targetRole === APP_ROLES.SUPERADMIN) return false;
  if (actorRole === APP_ROLES.SUPERADMIN) return true;
  if (actorRole !== APP_ROLES.ADMIN) return false;
  return targetRole !== APP_ROLES.ADMIN;
}

export function canAssignRole(
  actor: Pick<UserProfile, 'role'> | null | undefined,
  targetRole: string,
): boolean {
  if (!actor) return false;
  const actorRole = normalizeUserRole(actor.role);
  const normalizedTargetRole = normalizeUserRole(targetRole);

  if (normalizedTargetRole === APP_ROLES.SUPERADMIN) return false;
  if (actorRole === APP_ROLES.SUPERADMIN) return true;
  if (actorRole !== APP_ROLES.ADMIN) return false;
  return normalizedTargetRole !== APP_ROLES.ADMIN;
}
