import React from 'react';
import { ChevronRight } from 'lucide-react';
import { UiText } from '../../../components/ui/atoms/UiText';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';
import type { UserProfile } from '../useWelcomeLogic';
import { ROLE_STYLE_MAP } from '../constants';

interface ProfileListProps {
  profiles: UserProfile[];
  loading: boolean;
  onSelectUser: (user: UserProfile) => void;
}

export const ProfileList: React.FC<ProfileListProps> = ({ profiles, loading, onSelectUser }) => {
  const { t } = useI18n();

  if (loading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-[52px] w-full animate-pulse rounded-[18px] bg-[color-mix(in_srgb,var(--app-text)_5%,transparent)]"
          />
        ))}
      </div>
    );
  }

  if (profiles.length === 0) {
    return (
      <UiText as="div" size="sm" tone="secondary" className="py-8 text-center">
        {t('welcome.empty')}
      </UiText>
    );
  }

  return (
    <div className="space-y-2">
      <UiText
        as="p"
        size="xs"
        weight="semibold"
        tone="muted"
        heading
        className="ml-1 uppercase tracking-[0.18em]"
      >
        {t('welcome.employee')}
      </UiText>

      <div className="space-y-2">
        {profiles.map((user) => {
          const config = ROLE_STYLE_MAP[user.role?.toUpperCase() as keyof typeof ROLE_STYLE_MAP];

          return (
            <button
              key={user.id}
              type="button"
              onClick={() => onSelectUser(user)}
              className="group flex h-[52px] w-full items-center justify-between gap-3 rounded-[18px] border border-[var(--app-border)] bg-[var(--app-surface-raised)] px-4 text-left transition-all hover:border-[color-mix(in_srgb,var(--app-accent)_35%,transparent)] hover:bg-[var(--app-surface-raised)] hover:shadow-[var(--app-shadow-sm)]"
            >
              <div className="flex min-w-0 items-center gap-3">
                <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-[color-mix(in_srgb,var(--app-text)_6%,transparent)] text-[length:var(--text-xs)] font-semibold leading-[var(--leading-heading)] text-[var(--app-text-secondary)]">
                  {user.full_name.charAt(0).toUpperCase()}
                </div>
                <span className="truncate text-[length:var(--text-sm)] font-semibold leading-[var(--leading-body)] text-[var(--app-text)]">
                  {user.full_name}
                </span>
                {config ? (
                  <span
                    className={`flex shrink-0 items-center gap-1 rounded-full px-2 py-1 text-[length:var(--text-xs)] font-semibold leading-[var(--leading-heading)] ${config.className}`}
                  >
                    <config.Icon size={12} strokeWidth={2.3} />
                    {t(`role.${user.role?.toUpperCase()}` as TranslationKey)}
                  </span>
                ) : null}
              </div>
              <ChevronRight
                size={16}
                strokeWidth={2}
                className="shrink-0 text-[var(--app-text-muted)] transition-transform group-hover:translate-x-0.5"
              />
            </button>
          );
        })}
      </div>
    </div>
  );
};
