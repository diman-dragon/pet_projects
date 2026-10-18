// src/services/license/providers/types.ts

import type { LicensePayload } from '../config';

export interface ActivationProvider {
  activate(licenseKey: string): Promise<LicensePayload>;
  validate(stored: LicensePayload | null): Promise<LicensePayload>;
}
