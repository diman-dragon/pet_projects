// src/services/init/steps/ConfigStep.ts
// Ответственность: прочитать сохранённый путь к рабочей папке и убедиться, что она существует.
// Возвращает status 'setup' если конфига нет — это не ошибка, это нормальный первый запуск.

import { exists } from '@tauri-apps/plugin-fs';
import { ConfigService } from '../../config/ConfigService';
import { createLogger } from '../../logger/Logger';
import { InitStep, InitContext } from '../types';

const log = createLogger('ConfigStep');

export class ConfigStep implements InitStep {
  readonly name = 'Config';

  async run(ctx: InitContext): Promise<InitContext> {
    const rootPath = await ConfigService.load();

    if (!rootPath) {
      log.warn('Конфиг не найден → требуется первичная настройка.');
      return { ...ctx, status: 'setup' };
    }

    const physicalExists = await exists(rootPath);
    if (!physicalExists) {
      log.warn('Папка не существует на диске → требуется первичная настройка.');
      return { ...ctx, status: 'setup' };
    }

    log.info('OK, путь:', rootPath);
    return { ...ctx, rootPath };
  }
}
