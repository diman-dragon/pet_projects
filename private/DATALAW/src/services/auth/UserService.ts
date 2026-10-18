import { join } from '@tauri-apps/api/path';
import { exists, mkdir } from '@tauri-apps/plugin-fs';
import { AuditService } from '../audit/AuditService';
import { ConnectionManager } from '../db/ConnectionManager';
import { AppPaths } from '../config/AppPaths';
import { AppInitializer } from '../init/AppInitializer';
import type { UserProfile } from '../../types/entities';
import { CryptoService } from './CryptoService';
import { APP_ROLES, canAssignRole, canManageTargetUser, normalizeUserRole } from './permissions';
import { isProtectedSuperadmin } from './userGuards';

export interface UserRecord extends UserProfile {
  password_hash: string;
  salt: string;
  created_at?: string;
  last_login?: string | null;
}

export interface CreateUserInput {
  full_name: string;
  login: string;
  password: string;
  role: string;
}

export interface UpdateUserInput {
  id: number;
  full_name: string;
  role: string;
  password?: string;
}

const DB_KEY = 'user_system';

async function getDbOrNull() {
  if (ConnectionManager.exists(DB_KEY)) {
    return ConnectionManager.get(DB_KEY);
  }

  try {
    await AppInitializer.boot();
  } catch {
    // Ignore bootstrap errors and fall back to the direct path resolution below.
  }

  if (ConnectionManager.exists(DB_KEY)) {
    return ConnectionManager.get(DB_KEY);
  }

  const rootPath = AppPaths.getRootPath();
  if (!rootPath) {
    return null;
  }

  try {
    const userDbFolder = await join(rootPath, '.db_user');
    const dbPath = await join(userDbFolder, 'users.db');

    if (!(await exists(userDbFolder))) {
      await mkdir(userDbFolder, { recursive: true });
    }

    return await ConnectionManager.open(DB_KEY, dbPath);
  } catch {
    return null;
  }
}

async function countSuperadmins(): Promise<number> {
  const db = await getDbOrNull();
  if (!db) return 0;

  const rows = await db.select<{ count: number }[]>(
    'SELECT COUNT(*) as count FROM users WHERE role = $1',
    [APP_ROLES.SUPERADMIN],
  );
  return rows[0]?.count ?? 0;
}

export const UserService = {
  async listProfiles(): Promise<UserProfile[]> {
    const db = await getDbOrNull();
    if (!db) return [];

    return db.select<UserProfile[]>(
      'SELECT id, login, full_name, role FROM users ORDER BY full_name ASC, id ASC',
    );
  },

  async listStaffProfiles(): Promise<UserProfile[]> {
    const db = await getDbOrNull();
    if (!db) return [];

    return db.select<UserProfile[]>(
      'SELECT id, login, full_name, role FROM users WHERE role != $1 ORDER BY full_name ASC, id ASC',
      [APP_ROLES.SUPERADMIN],
    );
  },

  async getById(id: number): Promise<UserRecord | null> {
    const db = await getDbOrNull();
    if (!db) return null;

    const [user] = await db.select<UserRecord[]>(
      'SELECT id, login, full_name, role, password_hash, salt, created_at, last_login FROM users WHERE id = $1',
      [id],
    );
    return user ?? null;
  },

  async verifyLogin(id: number, password: string): Promise<UserProfile | null> {
    const user = await this.getById(id);
    if (!user) return null;

    const isValid = await CryptoService.verify(password, user.salt, user.password_hash);
    if (!isValid) return null;

    const db = await getDbOrNull();
    if (!db) return null;

    await db.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = $1', [id]);
    await AuditService.log({
      actorId: user.id,
      actorLogin: user.login,
      actorRole: user.role,
      action: 'login',
      entityType: 'user',
      entityId: user.id,
      entityLabel: user.full_name,
      details: 'Успешный вход в систему',
    });

    return {
      id: user.id,
      login: user.login,
      full_name: user.full_name,
      role: user.role,
    };
  },

  async createInitialSuperadmin(input: Omit<CreateUserInput, 'role'>): Promise<void> {
    if ((await countSuperadmins()) > 0) {
      throw new Error('Суперадмин уже существует. Повторное создание запрещено.');
    }

    const db = await getDbOrNull();
    if (!db) {
      throw new Error('Пользовательская база данных недоступна. Повторите запуск приложения.');
    }

    const salt = CryptoService.generateSalt();
    const hash = await CryptoService.hashPassword(input.password, salt);
    await db.execute(
      `INSERT INTO users (login, full_name, role, password_hash, salt)
       VALUES ($1, $2, $3, $4, $5)`,
      [input.login.trim(), input.full_name.trim(), APP_ROLES.SUPERADMIN, hash, salt],
    );

    const [created] = await db.select<UserProfile[]>(
      'SELECT id, login, full_name, role FROM users WHERE login = $1',
      [input.login.trim()],
    );

    await AuditService.log({
      actorId: created?.id ?? null,
      actorLogin: created?.login ?? input.login.trim(),
      actorRole: created?.role ?? APP_ROLES.SUPERADMIN,
      action: 'genesis-create',
      entityType: 'user',
      entityId: created?.id ?? null,
      entityLabel: input.full_name.trim(),
      details: 'Создан первый суперадмин системы',
    });
  },

  async createUser(actor: UserProfile, input: CreateUserInput): Promise<void> {
    if (!canAssignRole(actor, input.role)) {
      throw new Error('Недостаточно прав для назначения этой роли');
    }

    const db = await getDbOrNull();
    if (!db) {
      throw new Error('Пользовательская база данных недоступна. Повторите запуск приложения.');
    }

    const salt = CryptoService.generateSalt();
    const hash = await CryptoService.hashPassword(input.password, salt);
    const role = normalizeUserRole(input.role);

    await db.execute(
      `INSERT INTO users (full_name, login, password_hash, salt, role)
       VALUES ($1, $2, $3, $4, $5)`,
      [input.full_name.trim(), input.login.trim(), hash, salt, role],
    );

    const [created] = await db.select<UserProfile[]>(
      'SELECT id, login, full_name, role FROM users WHERE login = $1',
      [input.login.trim()],
    );

    await AuditService.logForUser(actor, {
      action: 'create',
      entityType: 'user',
      entityId: created?.id ?? null,
      entityLabel: input.full_name.trim(),
      details: `Создан пользователь с ролью ${role}`,
    });
  },

  async updateUser(actor: UserProfile, input: UpdateUserInput): Promise<void> {
    const target = await this.getById(input.id);
    if (!target) throw new Error('Пользователь не найден');
    if (isProtectedSuperadmin(target)) {
      throw new Error('Суперадмин не редактируется через раздел сотрудников.');
    }
    if (!canManageTargetUser(actor, target)) {
      throw new Error('Недостаточно прав для редактирования этого пользователя');
    }
    if (!canAssignRole(actor, input.role)) {
      throw new Error('Недостаточно прав для назначения этой роли');
    }

    const db = await getDbOrNull();
    if (!db) {
      throw new Error('Пользовательская база данных недоступна. Повторите запуск приложения.');
    }

    const role = normalizeUserRole(input.role);

    if (input.password?.trim()) {
      const salt = CryptoService.generateSalt();
      const hash = await CryptoService.hashPassword(input.password.trim(), salt);
      await db.execute(
        `UPDATE users
         SET full_name = $1, role = $2, password_hash = $3, salt = $4
         WHERE id = $5`,
        [input.full_name.trim(), role, hash, salt, input.id],
      );
    } else {
      await db.execute(
        `UPDATE users
         SET full_name = $1, role = $2
         WHERE id = $3`,
        [input.full_name.trim(), role, input.id],
      );
    }

    await AuditService.logForUser(actor, {
      action: 'update',
      entityType: 'user',
      entityId: input.id,
      entityLabel: input.full_name.trim(),
      details: `Обновлен пользователь, новая роль ${role}`,
    });
  },

  async deleteUser(actor: UserProfile, id: number): Promise<void> {
    const target = await this.getById(id);
    if (!target) throw new Error('Пользователь не найден');
    if (actor.id === id) {
      throw new Error('Нельзя удалить самого себя.');
    }
    if (isProtectedSuperadmin(target)) {
      throw new Error('Суперадмин не удаляется через раздел сотрудников.');
    }
    if (!canManageTargetUser(actor, target)) {
      throw new Error('Недостаточно прав для удаления этого пользователя');
    }

    const db = await getDbOrNull();
    if (!db) {
      throw new Error('Пользовательская база данных недоступна. Повторите запуск приложения.');
    }

    await db.execute('DELETE FROM users WHERE id = $1', [id]);
    await AuditService.logForUser(actor, {
      action: 'delete',
      entityType: 'user',
      entityId: id,
      entityLabel: target.full_name,
      details: `Удален пользователь @${target.login}`,
    });
  },
};
