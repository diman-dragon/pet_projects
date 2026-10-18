// src/services/license/emptyLicense.ts

import { DEFAULT_FEATURES, type LicensePayload } from './config';

export function emptyLicense(): LicensePayload {
  return {
    valid: false,
    activated: false,
    key: null,
    tier: null,
    fingerprint: null,
    features: { ...DEFAULT_FEATURES },
    expiresAt: null,
    issuedAt: null,
    signature: null,
    isMaster: false,
  };
}
