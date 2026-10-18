import { FileSearch, LogOut, Settings } from 'lucide-react';
import { hasPermission } from '../../../services/auth/permissions';
import type { UserProfile } from '../../../types/entities';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import {
  APP_LANGUAGE_LABELS,
  SUPPORTED_APP_LANGUAGES,
  type AppLanguage,
} from '../../../i18n/config';
import { Button } from '../../ui/form/Button';
import { UiPanel } from '../../ui/atoms/UiPanel';
import { UiCaption } from '../../ui/atoms/UiCaption';
import { UiText } from '../../ui/atoms/UiText';

interface HeaderUserDropdownProps {
  user: UserProfile;
  onOpenAudit: () => void;
  onLogout: () => void;
  onClose: () => void;
  onNavigate: (route: string) => void;
  canOpenSettings: boolean;
}

export function HeaderUserDropdown({
  user,
  onOpenAudit,
  onLogout,
  onClose,
  onNavigate,
  canOpenSettings,
}: HeaderUserDropdownProps) {
  const { t, language, setLanguage } = useI18n();
  const canOpenAudit = hasPermission(user, 'audit:view');

  return (
    <div className="absolute right-0 top-[46px] z-[100] w-64 origin-top-right animate-in fade-in zoom-in-95 duration-100">
      <UiPanel padding="none" mica className="py-1.5 shadow-[0_12px_32px_rgba(0,0,0,0.1)]">
        <div className="mb-1 border-b border-black/[0.05] px-3 py-2.5">
          <UiCaption as="p" overline className="mb-1">
            {t('header.authAsCaption')}
          </UiCaption>
          <UiText as="p" size="xs" weight="semibold" tone="default" className="truncate font-mono">
            {user?.login || t('header.systemUser')}
          </UiText>
        </div>

        <div className="px-2 pb-2 pt-1">
          <UiPanel mica elevation="sm" padding="sm" className="border-black/[0.08] bg-black/[0.02]">
            <UiCaption as="div" overline className="mb-2 px-0.5 text-black/40">
              {t('header.appSettingsCaption')}
            </UiCaption>

            <div className="mb-2" role="group" aria-label={t('header.languageLabel')}>
              <UiText as="div" size="xs" weight="medium" tone="muted" className="mb-1.5 px-0.5">
                {t('header.languageLabel')}
              </UiText>
              <div className="flex gap-1">
                {SUPPORTED_APP_LANGUAGES.map((lng: AppLanguage) => {
                  const active = language === lng;
                  return (
                    <button
                      key={lng}
                      type="button"
                      aria-pressed={active}
                      title={APP_LANGUAGE_LABELS[lng]}
                      onClick={() => {
                        void setLanguage(lng);
                      }}
                      className={[
                        'min-w-0 flex-1 rounded-md border px-1.5 py-2 text-center text-[length:var(--text-xs)] font-semibold leading-[var(--leading-heading)] transition-colors',
                        active
                          ? 'border-[var(--app-accent)] bg-[var(--app-accent)] text-white shadow-sm'
                          : 'border-black/[0.08] bg-white text-black/60 hover:border-black/[0.15] hover:bg-black/[0.02]',
                      ].join(' ')}
                    >
                      {lng.toUpperCase()}
                    </button>
                  );
                })}
              </div>
            </div>

            {canOpenSettings && (
              <Button
                variant="ghost"
                size="sm"
                icon={<Settings size={16} strokeWidth={2} />}
                onClick={() => {
                  onNavigate('settings');
                  onClose();
                }}
                className="w-full justify-start text-[var(--app-accent)] hover:bg-[var(--app-accent)]/5"
              >
                {t('header.openSettingsPage')}
              </Button>
            )}
          </UiPanel>
        </div>

        <div className="px-1">
          {canOpenAudit && (
            <>
              <Button
                variant="ghost"
                size="sm"
                icon={<FileSearch size={16} strokeWidth={2} />}
                onClick={() => {
                  onOpenAudit();
                  onClose();
                }}
                className="w-full justify-start"
              >
                {t('header.auditLog')}
              </Button>

              <div className="mx-2 my-1 h-px bg-black/[0.05]" />
            </>
          )}

          <Button
            variant="ghost"
            size="sm"
            icon={<LogOut size={16} strokeWidth={2} />}
            onClick={onLogout}
            className="w-full justify-start text-[var(--app-danger)] hover:bg-[var(--app-danger)]/5"
          >
            {t('header.logout')}
          </Button>
        </div>
      </UiPanel>
    </div>
  );
}
