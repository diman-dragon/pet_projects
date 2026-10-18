import { useState, useEffect } from 'react';
import { aiApi, SidecarStatus } from '../services/backend/aiApi';
import { createLogger } from '../services/logger/Logger';

export function useSidecarStatus() {
  const log = createLogger('useSidecarStatus');
  const [status, setStatus] = useState<SidecarStatus>({ running: false });
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    try {
      const current = await aiApi.getStatus();
      setStatus(current);
    } catch (err) {
      log.error('Failed to get sidecar status:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 10000);
    return () => clearInterval(interval);
  }, []);

  return { status, loading, refresh };
}
