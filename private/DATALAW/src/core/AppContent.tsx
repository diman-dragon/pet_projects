import { ErrorBoundary } from '../components/error/ErrorBoundary';
import Shell from '../components/layout/Shell/Shell';
import { useNavigation } from '../navigation/useNavigation';
import type { NavigationID } from '../pages/clients/types';
import { AppRouter } from '../routes/AppRouter';
import { ErrorReporter } from '../services/errors/ErrorReporter';
import { useLicenseStore } from '../store/useLicenseStore';
import type { AppState } from './hooks/useAppState';
import { useEffect } from 'react';
import { subscribeAppNavigation } from '../navigation/AppNavigationBus';
import { AppInitializer } from '../services/init/AppInitializer';

export default function AppContent({ appState }: { appState: AppState }) {
  const { handleNavigate } = useNavigation(appState);
  const license = useLicenseStore((state) => state.license);

  useEffect(() => {
    ErrorReporter.installGlobalHandlers();
  }, []);

  useEffect(() => subscribeAppNavigation(({ route, params }) => handleNavigate(route, params)), [
    handleNavigate,
  ]);

  const toNum = (id: NavigationID): number | null =>
    id === null || id === undefined ? null : Number(id);

  const systemServices = {
    clients: license?.features.clients ?? true,
    cases: license?.features.clients ?? true,
    knowledge: license?.features.knowledge ?? false,
    users: license?.features.users ?? false,
    settings: license?.features.settings ?? true,
    audit: license?.features.audit ?? false,
    laila: license?.features.laila ?? false,
  };

  return (
    <Shell
      user={appState.currentUser!}
      currentStatus={appState.status}
      activeServices={systemServices}
      onNavigate={handleNavigate}
      activeClientId={appState.activeClientId}
      activeCaseId={appState.activeCaseId}
      activeLineId={appState.activeLineId}
      activeAppealId={appState.activeAppealId}
      onOpenAudit={() => handleNavigate('audit')}
      onLogout={() => {
        void AppInitializer.lock();
        appState.setCurrentUser(null);
        appState.setStatus('locked');
      }}
    >
      <ErrorBoundary>
        <AppRouter
          status={appState.status}
          currentUser={appState.currentUser}
          activeClientId={appState.activeClientId}
          setStatus={appState.setStatus}
          setActiveClientId={(id: NavigationID) => appState.setActiveClientId(toNum(id))}
          activeCaseId={appState.activeCaseId}
          setActiveCaseId={(id: NavigationID) => appState.setActiveCaseId(toNum(id))}
          activeLineId={appState.activeLineId}
          setActiveLineId={(id: NavigationID) => appState.setActiveLineId(toNum(id))}
          activeAppealId={appState.activeAppealId}
          setActiveAppealId={(id: NavigationID) => appState.setActiveAppealId(toNum(id))}
        />
      </ErrorBoundary>
    </Shell>
  );
}
