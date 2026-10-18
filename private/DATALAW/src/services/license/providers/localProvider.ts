import { emptyLicense } from '../emptyLicense';
import { MachineFingerprintService } from '../MachineFingerprint';
import { TIER_FEATURES, type LicensePayload } from '../config';
import type { ActivationProvider } from './types';

async function buildMasterLicense(masterKey: string): Promise<LicensePayload> {
  return {
    valid: true,
    activated: true,
    key: masterKey,
    activationKey: null,
    tier: 'enterprise',
    fingerprint: 'MASTER',
    features: TIER_FEATURES.enterprise,
    expiresAt: null,
    issuedAt: new Date().toISOString(),
    signature: null,
    isMaster: true,
  };
}

function isExpired(expiresAt: string | null): boolean {
  return Boolean(expiresAt && Date.parse(expiresAt) <= Date.now());
}

async function buildLocalLicense(
  activationKey: string,
  fingerprint: string,
): Promise<LicensePayload> {
  const issued = await MachineFingerprintService.verifyIssuedLicenseKey(activationKey);
  if (isExpired(issued.expiresAt)) {
    throw new Error('Срок действия лицензионного ключа истек.');
  }

  return {
    valid: true,
    activated: true,
    key: issued.key,
    activationKey,
    tier: issued.tier,
    fingerprint,
    features: TIER_FEATURES[issued.tier],
    expiresAt: issued.expiresAt,
    issuedAt: issued.issuedAt,
    signature: null,
    isMaster: false,
  };
}

async function validateMaster(stored: LicensePayload): Promise<LicensePayload> {
  const isMasterKeyValid = await MachineFingerprintService.verifyMasterKey(stored.key ?? '');
  if (!isMasterKeyValid) return emptyLicense();
  return { ...stored, valid: true, activated: true };
}

async function validateLocal(stored: LicensePayload, fingerprint: string): Promise<LicensePayload> {
  if (
    !stored.key ||
    !stored.activationKey ||
    !stored.tier ||
    !stored.fingerprint ||
    !stored.issuedAt
  ) {
    return emptyLicense();
  }

  if (stored.fingerprint !== fingerprint || isExpired(stored.expiresAt)) {
    return emptyLicense();
  }

  try {
    const issued = await MachineFingerprintService.verifyIssuedLicenseKey(stored.activationKey);

    // We compare tier and key. For compact keys, we ignore strict issuedAt string comparison
    // as it can have different formats or stable values.
    const samePayload =
      issued.key === stored.key &&
      issued.tier === stored.tier &&
      (issued.expiresAt === stored.expiresAt ||
        (issued.expiresAt &&
          stored.expiresAt &&
          Date.parse(issued.expiresAt) === Date.parse(stored.expiresAt)));

    if (!samePayload) {
      return emptyLicense();
    }

    return { ...stored, valid: true, activated: true };
  } catch {
    return emptyLicense();
  }
}

export const localActivationProvider: ActivationProvider = {
  async activate(licenseKey: string): Promise<LicensePayload> {
    const normalized = licenseKey.trim();
    if (!normalized) {
      throw new Error('Введите лицензионный ключ.');
    }

    if (await MachineFingerprintService.verifyMasterKey(normalized)) {
      return buildMasterLicense(normalized);
    }

    const machine = await MachineFingerprintService.get();
    return buildLocalLicense(normalized, machine.fingerprint);
  },

  async validate(stored: LicensePayload | null): Promise<LicensePayload> {
    if (!stored?.key) return emptyLicense();

    if (stored.isMaster) {
      return validateMaster(stored);
    }

    const machine = await MachineFingerprintService.get();
    return validateLocal(stored, machine.fingerprint);
  },
};
