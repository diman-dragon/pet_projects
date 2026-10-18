import { useCallback, useEffect, useState } from 'react';
import { useSystemInit } from '../../hooks/useSystemInit';
import { LicenseService } from '../../services/license/LicenseService';
import { useLicenseStore } from '../../store/useLicenseStore';
import { useOllamaStore } from '../../store/ollamaStore';
import { createLogger } from '../../services/logger/Logger';
import type { UserProfile } from '../../types/entities';

export type AppState = ReturnType<typeof useAppState>;

export function useAppState() {
  const log = createLogger('useAppState');
  const system = useSystemInit();
  const setLicense = useLicenseStore((state) => state.setLicense);

  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
  const [status, setStatus] = useState<string>('checking');
  const [dbReady, setDbReady] = useState(false);

  const [activeClientId, setActiveClientId] = useState<number | string | null>(null);
  const [activeCaseId, setActiveCaseId] = useState<number | string | null>(null);
  const [activeLineId, setActiveLineId] = useState<number | string | null>(null);
  const [activeAppealId, setActiveAppealId] = useState<number | string | null>(null);

  // FIX: убраны useClientsLogic() и useUsersLogic() из этого хука.
  // Оба вызывали invoke() сразу при монтировании — до завершения AppInitializer.boot()
  // и открытия соединений с БД. Это приводило к race condition и unhandled rejections.
  // ClientsPage и UsersPage управляют своими экземплярами хуков самостоятельно
  // и монтируются только после того, как status переходит в 'clients'/'users',
  // то есть уже после полной инициализации системы.

  const onNavigate = useCallback((newStatus: string, params?: any) => {
    setStatus(newStatus);

    if (newStatus !== 'clients') {
      setActiveClientId(null);
      setActiveCaseId(null);
      setActiveLineId(null);
      setActiveAppealId(null);
      return;
    }

    if (params) {
      setActiveClientId(params.clientId ?? null);
      setActiveCaseId(params.caseId ?? null);
      setActiveLineId(params.lineId ?? null);
      setActiveAppealId(params.appealId ?? null);
    } else {
      setActiveClientId(null);
      setActiveCaseId(null);
      setActiveLineId(null);
      setActiveAppealId(null);
    }
  }, []);

  useEffect(() => {
    system
      .init()
      .then(async () => {
        const license = await LicenseService.validate().catch(() => null);
        setLicense(license);
        setDbReady(true);

        // [1.1] Инициализация профиля машины для AI настроек
        const initProfile = useOllamaStore.getState().initMachineProfile;
        await initProfile().catch((err) => {
          log.warn('[APP_STATE] Machine profile detection failed:', err);
        });
      })
      .catch((error: Error) => {
        log.error('[APP_STATE] System initialization failed:', error);
        setStatus('error');
      });
  }, [setLicense, system.init]);

  useEffect(() => {
    if (system.status !== 'checking') {
      setStatus(system.status);
    }
  }, [system.status]);

  return {
    status,
    setStatus,
    onNavigate,
    system,
    dbReady,
    currentUser,
    setCurrentUser,
    activeClientId,
    setActiveClientId,
    activeCaseId,
    setActiveCaseId,
    activeLineId,
    setActiveLineId,
    activeAppealId,
    setActiveAppealId,
    rootPath: system.dataPath,
  };
}
