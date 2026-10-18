import { useState, useCallback, useEffect } from 'react';
import { listen } from '@tauri-apps/api/event';
import { useI18n } from '@/i18n/AppLanguageProvider';
import {
  businessApi,
  type ClientDossierProvider,
  type ClientDossierSnapshot,
} from '@/services/backend/businessApi';
import { formatAppError } from './utils';

export const useClientDossier = (clientId: number, taxId?: string) => {
  const { t } = useI18n();
  const [providers, setProviders] = useState<ClientDossierProvider[]>([]);
  const [snapshots, setSnapshots] = useState<ClientDossierSnapshot[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const loadDossier = useCallback(async () => {
    try {
      const payload = await businessApi.getClientDossier(clientId);
      setProviders(payload.providers);
      setSnapshots(payload.snapshots);
    } catch (err) {
      setError(formatAppError(err));
    }
  }, [clientId]);

  useEffect(() => {
    loadDossier();
  }, [loadDossier]);

  useEffect(() => {
    let canceled = false;
    let unlistenDone: (() => void) | undefined;
    let unlistenError: (() => void) | undefined;

    listen<{ client_id: number; snapshot_id: number }>('dossier-refreshed', (event) => {
      if (event.payload.client_id === clientId) {
        loadDossier();
        setLoading(false);
        setSuccess(t('dossier.sync.success' as any));
      }
    }).then((fn) => {
      if (canceled) {
        fn();
      } else {
        unlistenDone = fn;
      }
    });

    listen<{ client_id: number; error: string }>('dossier-refresh-error', (event) => {
      if (event.payload.client_id === clientId) {
        setError(`${t('dossier.sync.errorPrefix' as any)} ${event.payload.error}`);
        setLoading(false);
      }
    }).then((fn) => {
      if (canceled) {
        fn();
      } else {
        unlistenError = fn;
      }
    });

    return () => {
      canceled = true;
      unlistenDone?.();
      unlistenError?.();
    };
  }, [clientId, loadDossier, t]);

  const refreshDossier = async () => {
    if (!taxId) return;
    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const result = await businessApi.refreshClientDossier(clientId, taxId);
      setProviders(result.providers);
    } catch (err) {
      setError(formatAppError(err));
      setLoading(false);
    }
  };

  return {
    providers,
    snapshots,
    loading,
    error,
    success,
    refreshDossier,
    setError,
    setSuccess,
  };
};
