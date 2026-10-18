import { useEffect, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { normalizeUserRole, hasPermission } from '../../../services/auth/permissions';
import type { UserProfile } from '../../../types/entities';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';
import { UserAvatar } from '../../ui/data/UserAvatar';
import { UiText } from '../../ui/atoms/UiText';
import { HeaderUserDropdown } from './HeaderUserDropdown';

interface HeaderUserMenuProps {
  user: UserProfile;
  onOpenAudit: () => void;
  onLogout: () => void;
  onNavigate: (route: string, params?: Record<string, unknown>) => void;
  activeServices: Record<string, boolean>;
}

export function HeaderUserMenu({
  user,
  onOpenAudit,
  onLogout,
  onNavigate,
  activeServices,
}: HeaderUserMenuProps) {
  const { t } = useI18n();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const role = normalizeUserRole(user?.role);
  const roleKey = `audit.role.${role}` as TranslationKey;
  const roleLabel = t(roleKey);
  const displayRole = roleLabel === roleKey ? role : roleLabel;
  const canOpenSettings = (activeServices.settings ?? true) && hasPermission(user, 'settings:view');

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative flex min-w-[200px] items-center justify-end gap-1" ref={menuRef}>
      <div className="flex min-w-0 flex-col items-end pr-1 leading-none">
        <UiText
          as="span"
          size="xs"
          weight="semibold"
          tone="default"
          className="max-w-[160px] truncate"
        >
          {user?.full_name || t('header.userFallback')}
        </UiText>
        <UiText as="span" size="xs" tone="muted" className="mt-1 max-w-[160px] truncate">
          {displayRole}
        </UiText>
      </div>

      <button
        type="button"
        aria-expanded={isOpen}
        aria-haspopup="menu"
        onClick={() => setIsOpen((value) => !value)}
        className={[
          'flex items-center gap-1.5 rounded-md border p-0.5 pl-2 transition-all duration-75',
          isOpen
            ? 'border-[var(--app-border-strong)] bg-[color-mix(in_srgb,var(--app-text)_4%,transparent)]'
            : 'border-[var(--app-border)] bg-[var(--app-surface-raised)] hover:border-[var(--app-border-strong)] hover:shadow-[var(--app-shadow-sm)]',
        ].join(' ')}
      >
        <UserAvatar name={user?.full_name} size="sm" />
        <ChevronDown
          className={[
            'mr-1.5 h-3 w-3 text-[var(--app-text-muted)] transition-transform duration-200',
            isOpen ? 'rotate-180' : '',
          ].join(' ')}
        />
      </button>

      {isOpen && (
        <HeaderUserDropdown
          user={user}
          onOpenAudit={onOpenAudit}
          onLogout={onLogout}
          onClose={() => setIsOpen(false)}
          onNavigate={(route) => onNavigate(route)}
          canOpenSettings={canOpenSettings}
        />
      )}
    </div>
  );
}
