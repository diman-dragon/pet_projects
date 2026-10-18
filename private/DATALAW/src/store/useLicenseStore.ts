// src/store/useLicenseStore.ts

import { create } from 'zustand';
import {
  DEFAULT_FEATURES,
  type FeatureName,
  type LicensePayload,
} from '../services/license/config';

interface LicenseStoreState {
  license: LicensePayload | null;
  setLicense: (license: LicensePayload | null) => void;
  hasFeature: (feature: FeatureName) => boolean;
  isValid: () => boolean;
}

export const useLicenseStore = create<LicenseStoreState>((set, get) => ({
  license: null,

  setLicense: (license) => set({ license }),

  hasFeature: (feature) => get().license?.features?.[feature] ?? DEFAULT_FEATURES[feature],

  isValid: () => get().license?.valid === true,
}));
