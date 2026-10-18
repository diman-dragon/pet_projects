export type LicenseTier = 'basic' | 'pro' | 'enterprise';
export type FeatureName =
  | 'clients'
  | 'knowledge'
  | 'rag'
  | 'users'
  | 'audit'
  | 'laila'
  | 'settings';
export type LicenseActivationMode = 'local' | 'server';

export interface FeatureSet {
  clients: boolean;
  knowledge: boolean;
  rag: boolean;
  users: boolean;
  audit: boolean;
  laila: boolean;
  settings: boolean;
}

export interface LicensePayload {
  valid: boolean;
  activated: boolean;
  key: string | null;
  activationKey?: string | null;
  tier: LicenseTier | null;
  fingerprint: string | null;
  features: FeatureSet;
  expiresAt: string | null;
  issuedAt: string | null;
  signature: string | null;
  isMaster?: boolean;
}

export const LICENSE_ACTIVATION_MODE: LicenseActivationMode = 'local';
export const LICENSE_FILE = 'license.json';

export const DEFAULT_FEATURES: FeatureSet = {
  clients: true,
  knowledge: false,
  rag: false,
  users: false,
  audit: false,
  laila: false,
  settings: true,
};

export const TIER_FEATURES: Record<LicenseTier, FeatureSet> = {
  basic: {
    clients: true,
    knowledge: true,
    rag: true,
    users: false,
    audit: false,
    laila: false,
    settings: true,
  },
  pro: {
    clients: true,
    knowledge: true,
    rag: true,
    users: true,
    audit: true,
    laila: true,
    settings: true,
  },
  enterprise: {
    clients: true,
    knowledge: true,
    rag: true,
    users: true,
    audit: true,
    laila: true,
    settings: true,
  },
};
