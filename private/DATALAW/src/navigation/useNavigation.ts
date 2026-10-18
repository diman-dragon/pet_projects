import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { normalizeRouteStatus, statusToPath } from './routes';

export interface NavigationAppState {
  setStatus: (s: string) => void;
  setActiveClientId: (id: number | null) => void;
  setActiveCaseId: (id: number | null) => void;
  setActiveLineId: (id: number | null) => void;
  setActiveAppealId: (id: number | null) => void;
}

export function useNavigation(appState: NavigationAppState) {
  const navigate = useNavigate();

  const { setStatus, setActiveClientId, setActiveCaseId, setActiveLineId, setActiveAppealId } =
    appState;

  const handleNavigate = useCallback(
    (
      route: string,
      params: {
        clientId?: number | null;
        caseId?: number | null;
        lineId?: number | null;
        appealId?: number | null;
      } = {},
    ) => {
      const nextRoute = normalizeRouteStatus(route);

      if (nextRoute !== 'clients') {
        setActiveClientId(null);
        setActiveCaseId(null);
        setActiveLineId(null);
        setActiveAppealId(null);
      } else {
        const { clientId = null, caseId = null, lineId = null, appealId = null } = params;
        setActiveClientId(clientId);
        setActiveCaseId(caseId);
        setActiveLineId(lineId);
        setActiveAppealId(appealId);
      }

      setStatus(nextRoute);
      navigate(statusToPath(nextRoute));
    },
    [navigate, setStatus, setActiveClientId, setActiveCaseId, setActiveLineId, setActiveAppealId],
  );

  return { handleNavigate };
}
