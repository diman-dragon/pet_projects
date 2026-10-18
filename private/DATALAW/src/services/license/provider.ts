// src/services/license/provider.ts

import { LICENSE_ACTIVATION_MODE } from './config';
import { localActivationProvider } from './providers/localProvider';
import { serverActivationProvider } from './providers/serverProvider';
import type { ActivationProvider } from './providers/types';

export function getActivationProvider(): ActivationProvider {
  return LICENSE_ACTIVATION_MODE === 'server' ? serverActivationProvider : localActivationProvider;
}
