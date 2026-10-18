import type { UserRole } from '../../types/entities';

export type { UserRole };

export interface User {
  id: string | number;
  full_name: string;
  role: UserRole;
  login: string;
  password?: string;
  email?: string;
  created_at?: string;
  _isEdit?: boolean;
}

export type StaffMember = User;
