import { UiText } from '../../ui/atoms/UiText';
import { HeaderLogo } from './HeaderLogo';
import { HeaderSearch } from './HeaderSearch';
import { HeaderUserMenu } from './HeaderUserMenu';
import { SystematizerIndicator } from './SystematizerIndicator';
import { IndexingIndicator } from './IndexingIndicator';
import type { UserProfile } from '../../../types/entities';

interface HeaderProps {
  user: UserProfile;
  onLogout: () => void;
  onOpenAudit: () => void;
  onNavigate: (route: string, params?: Record<string, unknown>) => void;
  activeServices: Record<string, boolean>;
}

export default function Header({
  user,
  onLogout,
  onOpenAudit,
  onNavigate,
  activeServices,
}: HeaderProps) {
  return (
    <div className="z-30 shrink-0 px-3 pt-3 select-none">
      <header className="flex h-14 w-full items-center justify-between rounded-[var(--app-radius-md)] border border-[var(--app-border)] bg-[var(--app-panel-bg)] px-5 shadow-[var(--app-shadow-sm)] backdrop-blur-xl">
        <div className="flex items-center gap-8">
          <HeaderLogo />
          <UiText as="span" size="xs" weight="bold" tone="muted" className="label-caps hidden xl:inline-flex">
            Dorlane Workspace
          </UiText>
        </div>
        <div className="flex-1 px-4 lg:px-8">
          <HeaderSearch user={user} onNavigate={onNavigate} />
        </div>
        <div className="flex items-center gap-4">
          <IndexingIndicator />
          <SystematizerIndicator />
          <HeaderUserMenu
            user={user}
            onOpenAudit={onOpenAudit}
            onLogout={onLogout}
            onNavigate={onNavigate}
            activeServices={activeServices}
          />
        </div>
      </header>
    </div>
  );
}
