// src/services/license/MachineFingerprint.ts

import { invoke } from '@tauri-apps/api/core';
import type { LicenseTier } from './config';

export interface MachineFingerprint {
  fingerprint: string;
  sources: string[];
}

export interface IssuedLicenseTokenPayload {
  key: string;
  tier: LicenseTier;
  expiresAt: string | null;
  issuedAt: string;
}

export const MachineFingerprintService = {
  get(): Promise<MachineFingerprint> {
    return invoke<MachineFingerprint>('get_machine_fingerprint');
  },

  verifyMasterKey(input: string): Promise<boolean> {
    return invoke<boolean>('verify_master_key', { input });
  },

  verifyIssuedLicenseKey(licenseKey: string): Promise<IssuedLicenseTokenPayload> {
    return invoke<IssuedLicenseTokenPayload>('verify_license_key', { licenseKey });
  },
};
