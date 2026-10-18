// src/services/init/steps/ValidationStep.ts
// Ответственность: убедиться, что все соединения открыты и система готова к работе.
// Финальный сторожевой шаг пайплайна boot.

import { ConnectionManager } from '../../db/ConnectionManager';
import { createLogger } from '../../logger/Logger';
import { InitStep, InitContext } from '../types';

const log = createLogger('ValidationStep');

const REQUIRED_CONNECTIONS = ['user_system', 'kb_main', 'clients_main'] as const;

export class ValidationStep implements InitStep {
  readonly name = 'Validation';

  async run(ctx: InitContext): Promise<InitContext> {
    for (const key of REQUIRED_CONNECTIONS) {
      if (!ConnectionManager.exists(key)) {
        throw new Error(
          `Соединение не найдено: [${key}]. Возможно, DatabaseConnectStep не отработал.`,
        );
      }
    }

    log.info('✅ Система готова к работе.');
    return ctx;
  }
}
