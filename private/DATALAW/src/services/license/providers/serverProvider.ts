// src/services/license/providers/serverProvider.ts
//
// Провайдер серверной активации.
//
// Когда будет готов бэкенд:
//   1. POST /api/v1/licenses/activate  { key, fingerprint }
//      → { payload: LicensePayload, signature: string }
//   2. POST /api/v1/licenses/validate  { key, fingerprint }
//      → { payload: LicensePayload, signature: string }
//
// Подпись на сервере должна использовать ту же схему, что и локальная проверка ключей.
// При переходе на серверную лицензию локальную verify-проверку можно заменить ответом сервера.
//
// Для переключения: установить LICENSE_ACTIVATION_MODE = 'server' в config.ts.

import { emptyLicense } from '../emptyLicense';
import { MachineFingerprintService } from '../MachineFingerprint';
import type { LicensePayload } from '../config';
import type { ActivationProvider } from './types';

const SERVER_URL = import.meta.env.VITE_LICENSE_SERVER_URL ?? 'https://license.Dorlane.app';

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${SERVER_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const msg = await res.text().catch(() => 'Unknown error');
    throw new Error(`License server error (${res.status}): ${msg}`);
  }

  return res.json() as Promise<T>;
}

export const serverActivationProvider: ActivationProvider = {
  async activate(licenseKey: string): Promise<LicensePayload> {
    const machine = await MachineFingerprintService.get();

    const payload = await post<LicensePayload>('/api/v1/licenses/activate', {
      key: licenseKey.trim().toUpperCase(),
      fingerprint: machine.fingerprint,
    });

    return payload;
  },

  async validate(stored: LicensePayload | null): Promise<LicensePayload> {
    if (!stored?.key) return emptyLicense();

    try {
      const machine = await MachineFingerprintService.get();
      const payload = await post<LicensePayload>('/api/v1/licenses/validate', {
        key: stored.key,
        fingerprint: machine.fingerprint,
      });
      return payload;
    } catch {
      // При недоступности сервера — fallback на сохранённую лицензию с проверкой подписи.
      // Локальный провайдер импортируем динамически, чтобы не тащить круговую зависимость.
      const { localActivationProvider } = await import('./localProvider');
      return localActivationProvider.validate(stored);
    }
  },
};
