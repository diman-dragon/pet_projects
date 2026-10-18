import { MasterKeyService } from '../../security/MasterKeyService';
import { createLogger } from '../../logger/Logger';
import type { InitContext, InitStep } from '../types';
import { join } from '@tauri-apps/api/path';

const log = createLogger('MasterKey');

export class MasterKeyStep implements InitStep {
  readonly name = 'MasterKey';

  async run(ctx: InitContext): Promise<InitContext> {
    if (ctx.paths?.rootPath) {
      MasterKeyService.setStatusDbPath(await join(ctx.paths.rootPath, '.db_user', 'users.db'));
    }

    if (MasterKeyService.isUnlocked()) {
      return { ...ctx, masterKeyReady: true };
    }

    if (ctx.masterPassword?.trim()) {
      await MasterKeyService.unlockWithPassword(ctx.masterPassword);
      log.info('Master key is available in process memory');
      return { ...ctx, masterKeyReady: true, masterPassword: undefined };
    }

    const env = (import.meta as ImportMeta & { env?: Record<string, string | boolean | undefined> }).env;
    const envPassword = env?.VITE_Dorlane_MASTER_PASSWORD as string | undefined;
    if (envPassword?.trim()) {
      await MasterKeyService.unlockWithPassword(envPassword);
      log.info('Master key is available in process memory');
      return { ...ctx, masterKeyReady: true };
    }

    log.info('Master key is locked, proceeding unlocked');
    return { ...ctx, masterKeyReady: true };
  }
}
