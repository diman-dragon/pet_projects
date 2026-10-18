import type { ReactNode } from 'react';
import { PageLoader } from '../components/ui';
import { useI18n } from '../i18n/AppLanguageProvider';
import ActivationPage from '../pages/license/ActivationPage';
import SetupPage from '../pages/setup/SetupPage';
import UnlockPage from '../pages/security/UnlockPage';
import WelcomePage from '../pages/welcome/WelcomePage';
import type { AppState } from './hooks/useAppState';

interface AppGuardsProps {
  children: ReactNode;
  appState: AppState;
}

export default function AppGuards({ children, appState }: AppGuardsProps) {
  const { t } = useI18n();
  const { status, currentUser, setCurrentUser, setStatus } = appState;

  if (status === 'checking') {
    return <PageLoader text={t('app.checking')} />;
  }

  if (status === 'setup') {
    return <SetupPage onComplete={() => window.location.reload()} />;
  }

  if (status === 'license') {
    return <ActivationPage onActivated={() => window.location.reload()} />;
  }

  if (status === 'locked') {
    return <UnlockPage onUnlocked={(nextStatus) => setStatus(nextStatus)} />;
  }

  if (!currentUser) {
    return (
      <WelcomePage
        onSelect={(user) => {
          setCurrentUser(user);
          setStatus('dashboard');
        }}
        onChangeWorkspace={() => setStatus('setup')}
      />
    );
  }

  return <>{children}</>;
}
