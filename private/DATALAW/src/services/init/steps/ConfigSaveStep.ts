// src/services/init/steps/ConfigSaveStep.ts
// Ответственность: сохранить rootPath в конфиг приложения.
// Используется только в пайплайне setup (первичная настройка).

import { ConfigService } from '../../config/ConfigService';
import { createLogger } from '../../logger/Logger';
import { InitStep, InitContext } from '../types';

const log = createLogger('ConfigSave');

export class ConfigSaveStep implements InitStep {
  readonly name = 'ConfigSave';

  async run(ctx: InitContext): Promise<InitContext> {
    await ConfigService.save(ctx.rootPath!);
    log.info('Конфиг сохранён:', ctx.rootPath);
    return ctx;
  }
}
