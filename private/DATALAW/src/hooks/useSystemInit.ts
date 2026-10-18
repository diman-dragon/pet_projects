import { useCallback, useState } from 'react';
import { AppPaths } from '../services/config/AppPaths';
import { AppInitializer } from '../services/init/AppInitializer';
import { createLogger } from '../services/logger/Logger';

const log = createLogger('useSystemInit');

export type SystemStatus = 'checking' | 'setup' | 'locked' | 'welcome' | 'license' | 'error' | string;

export function useSystemInit() {
  const [status, setStatus] = useState<SystemStatus>('checking');
  const [dataPath, setDataPath] = useState<string | null>(null);

  const init = useCallback(async (onReady?: () => Promise<void>) => {
    try {
      const result = await AppInitializer.boot();

      if (result.status === 'ready') {
        const rootPath = result.rootPath || null;
        setDataPath(rootPath);
        if (rootPath) AppPaths.setRootPath(rootPath);
        if (onReady) await onReady();
        setStatus('welcome');
        return;
      }

      if (result.status === 'setup') {
        setStatus('setup');
        return;
      }

      if (result.status === 'license') {
        log.warn('License activation required');
        setStatus('license');
        return;
      }

      if (result.status === 'locked') {
        if (result.rootPath) {
          setDataPath(result.rootPath);
          AppPaths.setRootPath(result.rootPath);
        }
        setStatus('locked');
        return;
      }

      setStatus('error');
    } catch (err) {
      log.error('[SYSTEM] Boot Failed with Exception:', err);
      setStatus('setup');
    }
  }, []);

  return { status, setStatus, dataPath, init };
}
