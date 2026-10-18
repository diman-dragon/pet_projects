import { useState, useEffect, useCallback } from 'react';
import { businessApi } from '@/services/backend/businessApi';
import type { CalendarValue, StatData } from './types';

export function useDashboardLogic() {
  const [date, setDate] = useState<CalendarValue>(new Date());
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<StatData>({
    docsCount: 0,
    clientsCount: 0,
    winRate: 0,
    pnpProgress: 0,
    clientDist: [],
    casesFlow: [],
    deadlines: [],
    recentLogs: [],
  });

  const loadData = useCallback(async () => {
    try {
      const nextStats = await businessApi.getDashboardStats();
      setStats(nextStats);
      setError(null);
      setIsReady(true);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('[Dashboard Load Error]:', err);
      setError(message);
    }
  }, []);

  useEffect(() => {
    const timer = setInterval(loadData, 5000);
    loadData();
    return () => clearInterval(timer);
  }, [loadData]);

  return { date, setDate, isReady, stats, error };
}
