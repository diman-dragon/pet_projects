// src/pages/users/components/UsersTable.tsx

import { BarChart3, Briefcase, Edit2, Shield, Trash2, User as UserIcon } from 'lucide-react';
import { Badge, UserAvatar } from '../../../components/ui';
import { AppTable } from '../../../components/ui/data/AppTable';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';
import type { User, UserRole } from '../types';

interface Props {
  items: User[];
  onEdit: (user: User) => void;
  onDelete: (user: User) => void;
  canEdit: (user: User) => boolean;
  canDelete: (user: User) => boolean;
}

const ROLE_STYLE: Record<
  UserRole,
  { variant: 'amber' | 'blue' | 'purple' | 'green'; icon: React.ReactNode }
> = {
  SUPERADMIN: { variant: 'amber', icon: <Shield size={10} /> },
  ADMIN: { variant: 'amber', icon: <Shield size={10} /> },
  LAWYER: { variant: 'blue', icon: <Briefcase size={10} /> },
  ASSISTANT: { variant: 'purple', icon: <UserIcon size={10} /> },
  ANALYST: { variant: 'green', icon: <BarChart3 size={10} /> },
};

export function UsersTable({ items, onEdit, onDelete, canEdit, canDelete }: Props) {
  const { t } = useI18n();

  const columns = [
    { key: 'id', label: 'ID', width: '6%' },
    { key: 'name', label: t('users.table.col.name'), width: '44%' },
    { key: 'login', label: t('users.table.col.login'), width: '16.666%' },
    { key: 'role', label: t('users.table.col.role'), width: '16.666%' },
    { key: 'actions', label: t('users.table.col.actions'), width: '16.666%' },
  ];

  return (
    <div className="w-full overflow-hidden">
      <AppTable columns={columns} elevated={false}>
        {items.map((user) => {
          const role = (user.role as string).toUpperCase() as UserRole;
          const style = ROLE_STYLE[role] ?? ROLE_STYLE.LAWYER;
          const editable = canEdit(user);
          const deletable = canDelete(user);
          const roleKey = `role.${role}` as TranslationKey;

          return (
            <tr
              key={user.id}
              className="group transition-colors hover:bg-[var(--app-surface-soft)]"
            >
              <td className="pl-6 pr-4 py-4">
                <span className="font-mono text-[11px] tracking-tighter text-[var(--app-text-muted)]">
                  {String(user.id).padStart(3, '0')}
                </span>
              </td>

              <td className="px-4 py-4">
                <div className="flex items-center gap-3 py-0.5">
                  <UserAvatar name={user.full_name} />
                  <span className="truncate text-[13px] font-semibold leading-none text-[var(--app-text)]">
                    {user.full_name}
                  </span>
                </div>
              </td>

              <td className="px-4 py-4">
                <span className="block truncate font-mono text-[12px] tracking-tight text-[var(--app-text-secondary)]">
                  @{user.login}
                </span>
              </td>

              <td className="px-4 py-4">
                <Badge label={t(roleKey)} variant={style.variant} icon={style.icon} />
              </td>

              <td className="px-4 py-4">
                <div className="flex items-center justify-start gap-1">
                  <button
                    onClick={() => editable && onEdit(user)}
                    disabled={!editable}
                    title={editable ? t('users.table.edit') : t('users.table.noPermission')}
                    className="flex h-8 w-8 items-center justify-center rounded-[10px] text-[var(--app-text-muted)] transition-all active:scale-90 hover:bg-[var(--app-surface-soft)] hover:text-[var(--app-text)] disabled:cursor-not-allowed disabled:opacity-30"
                  >
                    <Edit2 size={15} strokeWidth={2.2} />
                  </button>
                  <button
                    onClick={() => deletable && onDelete(user)}
                    disabled={!deletable}
                    title={deletable ? t('users.table.delete') : t('users.table.noPermission')}
                    className="flex h-8 w-8 items-center justify-center rounded-[10px] text-[var(--app-text-muted)] transition-all active:scale-90 hover:bg-[var(--app-danger)] hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
                  >
                    <Trash2 size={15} strokeWidth={2.2} />
                  </button>
                </div>
              </td>
            </tr>
          );
        })}
      </AppTable>

      {items.length === 0 && (
        <div className="py-12 text-center text-[13px] text-[var(--app-text-muted)]">
          {t('users.table.empty')}
        </div>
      )}
    </div>
  );
}
