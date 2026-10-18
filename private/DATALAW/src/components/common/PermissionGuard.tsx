import type { ReactNode } from 'react';
import type { AppPermission } from '../../services/auth/permissions';
import { hasPermission } from '../../services/auth/permissions';
import type { UserProfile } from '../../types/entities';

interface PermissionGuardProps {
  user: UserProfile | null;
  permission: AppPermission;
  fallback?: ReactNode;
  children: ReactNode;
}

export function PermissionGuard({
  user,
  permission,
  fallback = null,
  children,
}: PermissionGuardProps) {
  return hasPermission(user, permission) ? <>{children}</> : <>{fallback}</>;
}
