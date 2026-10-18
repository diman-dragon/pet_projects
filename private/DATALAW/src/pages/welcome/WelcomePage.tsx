import { useEffect, type ReactNode } from 'react';
import { KeyRound, LogOut, Shield, UserPlus, Users, X } from 'lucide-react';
import { DorlaneLogo } from '../../components/common/DorlaneLogo';
import { PublicLanguageSwitcher } from '../../components/auth/PublicLanguageSwitcher';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { Button } from '../../components/ui/form/Button';
import { UiText } from '../../components/ui/atoms/UiText';
import { AuthForm } from './components/AuthForm';
import { GenesisForm } from './components/GenesisForm';
import { ProfileList } from './components/ProfileList';
import { useWelcomeLogic, type UserProfile } from './useWelcomeLogic';
import { pluralEmployees } from './utils';

export type { UserProfile };

interface WelcomeProps {
  onSelect: (p: UserProfile) => void;
  onChangeWorkspace: () => void;
}

export default function WelcomePage({ onSelect, onChangeWorkspace }: WelcomeProps) {
  const logic = useWelcomeLogic(onSelect);
  const { language, t } = useI18n();

  const showAuth = !!logic.selectedUser && !logic.isGenesis;
  const showGenesis = logic.isGenesis;
  const showProfiles = !showAuth && !showGenesis;

  const currentTitle = showProfiles
    ? t('welcome.selectTitle')
    : showAuth
      ? t('welcome.authTitle')
      : t('welcome.genesisTitle');
  const currentDesc = showProfiles
    ? t('welcome.selectDesc')
    : showAuth
      ? t('welcome.authDesc', { name: logic.selectedUser!.full_name })
      : t('welcome.genesisDesc');

  const currentIcon: ReactNode = showProfiles ? (
    <Users size={18} strokeWidth={1.8} className="text-white" />
  ) : showAuth ? (
    <KeyRound size={18} strokeWidth={1.8} className="text-white" />
  ) : (
    <Shield size={18} strokeWidth={1.8} className="text-white" />
  );

  useEffect(() => {
    if (!showAuth) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') logic.setSelectedUser(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [showAuth, logic.setSelectedUser]);

  return (
    <div
      className="fixed inset-0 z-[1000] flex flex-col items-center justify-center gap-6 overflow-y-auto p-6"
      style={{ backgroundColor: 'var(--app-modal-scrim)' }}
    >
      <div className="pointer-events-none absolute inset-0 backdrop-blur-xl" aria-hidden />
      <div className="relative z-10 flex w-full max-w-[440px] flex-col items-center gap-5">
        <DorlaneLogo size="lg" showText className="gap-4" />
        <PublicLanguageSwitcher />
        <div
          className="w-full overflow-hidden rounded-[var(--app-radius-lg)] border border-[var(--app-border)] shadow-[var(--app-shadow-md)] backdrop-blur-xl backdrop-saturate-150"
          style={{ background: 'var(--app-panel-bg)' }}
        >
          <div className="flex min-h-[length:var(--app-modal-header-min-h)] items-center justify-between border-b border-[var(--app-border)] px-[var(--app-modal-padding-x)] py-[var(--app-modal-padding-y-header)]">
            <UiText as="div" size="md" weight="semibold" heading className="min-w-0 pr-2">
              {currentTitle}
            </UiText>
            {showAuth && (
              <button
                type="button"
                onClick={() => logic.setSelectedUser(null)}
                className="flex size-9 shrink-0 items-center justify-center rounded-full text-[var(--app-text-muted)] transition-all hover:bg-[color-mix(in_srgb,var(--app-text)_6%,transparent)] hover:text-[var(--app-text)]"
                aria-label={t('modal.close')}
              >
                <X size={18} strokeWidth={2} />
              </button>
            )}
          </div>
          <div className="max-h-[min(72vh,640px)] overflow-y-auto px-[var(--app-modal-padding-x)] py-[var(--app-modal-padding-y-body)] text-[length:var(--text-sm)] leading-[var(--leading-body)] text-[var(--app-text)] custom-scroll">
            <div className="flex flex-col gap-5">
              <div className="flex items-center gap-3 rounded-[20px] border border-[var(--app-border)] bg-[var(--app-surface-soft)] p-4">
                <div className="flex size-10 items-center justify-center rounded-[14px] bg-[var(--app-accent)] shadow-[var(--app-shadow-sm)]">
                  {currentIcon}
                </div>
                <div>
                  <UiText as="p" size="sm" weight="semibold" tone="default">
                    {currentTitle}
                  </UiText>
                  <UiText as="p" size="xs" tone="secondary" className="mt-1">
                    {currentDesc}
                  </UiText>
                </div>
              </div>
              <div className="flex min-h-[160px] flex-col justify-start">
                {showProfiles ? (
                  <ProfileList
                    profiles={logic.profiles}
                    loading={logic.loading}
                    onSelectUser={logic.setSelectedUser}
                  />
                ) : showAuth ? (
                  <AuthForm logic={logic} />
                ) : (
                  <GenesisForm logic={logic} />
                )}
              </div>
              {(showAuth || showGenesis) && (
                <Button
                  variant="primary"
                  size="md"
                  disabled={logic.isVerifying}
                  icon={showAuth ? <KeyRound size={14} /> : <UserPlus size={14} />}
                  onClick={() =>
                    (
                      document.getElementById(
                        showAuth ? 'auth-form' : 'gen-form',
                      ) as HTMLFormElement
                    )?.requestSubmit()
                  }
                >
                  {showAuth
                    ? logic.isVerifying
                      ? t('welcome.verify')
                      : t('welcome.signIn')
                    : t('welcome.createProfile')}
                </Button>
              )}
              {showProfiles && !logic.loading && logic.profiles.length > 0 && (
                <div className="text-center">
                  <UiText as="p" size="xs" tone="muted" className="leading-[var(--leading-body)]">
                    {logic.profiles.length}{' '}
                    {pluralEmployees(language as any, logic.profiles.length, t)}
                  </UiText>
                </div>
              )}
            </div>
          </div>
          <div className="flex w-full items-center justify-between gap-3 border-t border-[var(--app-border)] bg-[var(--app-surface-soft)] px-[var(--app-modal-padding-x)] py-[var(--app-modal-padding-y-footer)]">
            <button
              type="button"
              onClick={onChangeWorkspace}
              className="inline-flex items-center gap-2 text-[length:var(--text-xs)] font-semibold leading-[var(--leading-body)] text-[var(--app-text-secondary)] transition-colors hover:text-[var(--app-text)]"
            >
              <LogOut size={14} strokeWidth={2} />
              {t('welcome.changeWorkspace')}
            </button>
            <UiText
              as="span"
              size="xs"
              weight="semibold"
              tone="muted"
              heading
              className="uppercase tracking-[0.18em]"
            >
              {t('welcome.footerBrand')}
            </UiText>
          </div>
        </div>
      </div>
    </div>
  );
}
