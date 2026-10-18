// src/pages/audit/components/UsersTab.tsx

import { useMemo, useState } from 'react';
import { Calendar, Search, User } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';
import type { UserRecord } from '../types';

const ROLE_STYLES: Record<string, { bg: string; text: string }> = {
  SUPERADMIN: { bg: 'bg-red-50', text: 'text-red-600' },
  ADMIN: { bg: 'bg-amber-50', text: 'text-amber-700' },
  LAWYER: { bg: 'bg-blue-50', text: 'text-blue-600' },
  ASSISTANT: { bg: 'bg-green-50', text: 'text-green-600' },
  ANALYST: { bg: 'bg-fuchsia-50', text: 'text-fuchsia-700' },
};

function dateLocale(language: string): string {
  if (language === 'sr') return 'sr-Latn-RS';
  if (language === 'en') return 'en-US';
  return 'ru-RU';
}

export function UsersTab({ users }: { users: UserRecord[] }) {
  const { t, language } = useI18n();
  const loc = dateLocale(language);
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return users;
    return users.filter(
      (user) => user.full_name.toLowerCase().includes(q) || user.login.toLowerCase().includes(q),
    );
  }, [users, search]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4 rounded-xl border border-black/[0.06] bg-white p-3 shadow-sm">
        <div className="relative max-w-md flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-black/20" size={14} />
          <input
            type="text"
            placeholder={t('audit.users.search')}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-lg border border-black/[0.05] bg-black/[0.02] py-1.5 pl-9 pr-4 text-[length:var(--text-sm)] focus:outline-none"
          />
        </div>
        <div className="rounded-md bg-black/[0.03] px-3 py-1.5 text-[length:var(--text-xs)] font-mono uppercase text-black/30">
          {t('audit.users.total', { count: users.length })}
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-black/[0.08] bg-white shadow-sm">
        <table className="w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-black/[0.06] bg-black/[0.02]">
              <th className="w-16 px-6 py-3 text-[length:var(--text-xs)] font-bold uppercase tracking-wider text-black/40">
                {t('audit.users.colId')}
              </th>
              <th className="px-6 py-3 text-[length:var(--text-xs)] font-bold uppercase tracking-wider text-black/40">
                {t('audit.users.employee')}
              </th>
              <th className="px-6 py-3 text-[length:var(--text-xs)] font-bold uppercase tracking-wider text-black/40">
                {t('audit.users.login')}
              </th>
              <th className="px-6 py-3 text-[length:var(--text-xs)] font-bold uppercase tracking-wider text-black/40">
                {t('audit.users.access')}
              </th>
              <th className="px-6 py-3 text-right text-[length:var(--text-xs)] font-bold uppercase tracking-wider text-black/40">
                {t('audit.users.createdAt')}
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-black/[0.04]">
            {filtered.map((user) => {
              const roleKey = user.role.toUpperCase();
              const roleStyle = ROLE_STYLES[roleKey] ?? { bg: 'bg-gray-50', text: 'text-gray-500' };
              const roleLabel =
                roleKey in ROLE_STYLES ? t(`audit.role.${roleKey}` as TranslationKey) : user.role;

              return (
                <tr key={user.id} className="transition-colors hover:bg-black/[0.01]">
                  <td className="px-6 py-4 font-mono text-[length:var(--text-xs)] text-black/30">
                    #{user.id}
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="flex size-8 items-center justify-center rounded-full bg-black/[0.04] text-black/30">
                        <User size={14} />
                      </div>
                      <span className="text-[length:var(--text-sm)] font-bold text-black/80">
                        {user.full_name}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 font-mono text-[length:var(--text-xs)] text-black/40">
                    @{user.login}
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className={`inline-flex items-center rounded px-2 py-0.5 text-[length:var(--text-xs)] font-bold uppercase tracking-wider ${roleStyle.bg} ${roleStyle.text}`}
                    >
                      {roleLabel}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end gap-1.5 font-mono text-[length:var(--text-xs)] text-black/35">
                      <Calendar size={12} />
                      {user.created_at ? new Date(user.created_at).toLocaleDateString(loc) : '—'}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
