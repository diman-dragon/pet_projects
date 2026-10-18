import { join } from '@tauri-apps/api/path';
import { useOllamaStore } from '../../store/ollamaStore';
import { AppLanguageService } from '../config/AppLanguageService';
import { LicenseService } from '../license/LicenseService';
import { runSearchSchemaBootstrap } from '../search/searchSchemaBootstrap';
import { EmbeddingModelService } from '../ai/EmbeddingModelService';
import { MasterKeyService } from '../security/MasterKeyService';
import { ConnectionManager } from '../db/ConnectionManager';
import { runPipeline } from './pipeline';
import type { InitContext, InitResult } from './types';
import { ConfigSaveStep } from './steps/ConfigSaveStep';
import { ConfigStep } from './steps/ConfigStep';
import { DatabaseConnectStep } from './steps/DatabaseConnectStep';
import { DatabaseMigrationStep } from './steps/DatabaseMigrationStep';
import { DatabaseSchemaStep } from './steps/DatabaseSchemaStep';
import { FolderStep } from './steps/FolderStep';
import { MasterKeyStep } from './steps/MasterKeyStep';
import { SqliteVecStep } from './steps/SqliteVecStep';
import { ValidationStep } from './steps/ValidationStep';

const postUnlockSteps = [
  new DatabaseConnectStep(),
  new SqliteVecStep(),
  new DatabaseSchemaStep(),
  new DatabaseMigrationStep(),
  new ValidationStep(),
];

export const AppInitializer = {
  _bootPromise: null as Promise<InitResult> | null,
  _bootResult: null as InitResult | null,
  _lockedContext: null as InitContext | null,

  async boot(): Promise<InitResult> {
    if (this._bootResult) return this._bootResult;
    if (this._bootPromise) return this._bootPromise;

    this._bootPromise = runPipeline(
      [new ConfigStep(), new FolderStep(), new MasterKeyStep(), ...postUnlockSteps],
      { status: 'running' },
    )
      .then((ctx) => this.finishBoot(ctx))
      .finally(() => {
        this._bootPromise = null;
      });

    return this._bootPromise;
  },

  async setupWithPassword(userSelectedPath: string, masterPassword: string): Promise<string> {
    const rootPath = await join(userSelectedPath, 'Dorlane');

    const ctx = await runPipeline(
      [
        new FolderStep(),
        new MasterKeyStep(),
        new DatabaseConnectStep(),
        new SqliteVecStep(),
        new DatabaseSchemaStep(),
        new DatabaseMigrationStep(),
        new ConfigSaveStep(),
      ],
      { status: 'running', rootPath, masterPassword },
    );

    if (ctx.status === 'error' || ctx.status === 'locked') {
      throw new Error(ctx.error ?? 'Master password is required');
    }

    await AppLanguageService.get().catch(() => { });
    await runSearchSchemaBootstrap().catch(() => { });
    return ctx.rootPath!;
  },

  async unlock(masterPassword: string): Promise<InitResult> {
    await MasterKeyService.unlockWithPassword(masterPassword);

    if (!this._lockedContext) {
      this._bootResult = null;
      return this.boot();
    }

    const ctx = await runPipeline(postUnlockSteps, {
      ...this._lockedContext,
      status: 'running',
      masterKeyReady: true,
    });
    return this.finishBoot(ctx);
  },

  async lock(): Promise<void> {
    await ConnectionManager.closeAll();
    MasterKeyService.lock();
    this._bootResult = null;
  },

  async finishBoot(ctx: InitContext): Promise<InitResult> {
    if (ctx.status === 'locked') {
      this._lockedContext = ctx;
      return { status: 'locked', rootPath: ctx.rootPath };
    }

    if (ctx.status !== 'error' && ctx.status !== 'setup') {
      await AppLanguageService.get().catch(() => { });
    }

    if (ctx.status === 'running') {
      const license = await LicenseService.validate();
      if (!license.valid) {
        const result = { status: 'license' as const, rootPath: ctx.rootPath };
        this._bootResult = result;
        return result;
      }
    }

    const result = toInitResult(ctx);
    if (result.status === 'ready') {
      this._lockedContext = null;
      this._bootResult = result;
      this.startReadyBackgroundTasks(ctx);
    }
    return result;
  },

  startReadyBackgroundTasks(ctx: InitContext): void {
    runSearchSchemaBootstrap().catch(() => { });

    import('../search/GhostRecordService')
      .then(({ GhostRecordService }) => {
        GhostRecordService.check(false).catch(() => { });
      })
      .catch(() => { });

    if (ctx.paths?.kbPath) {
      import('../search/FileWatcherService')
        .then(({ FileWatcherService }) => {
          FileWatcherService.start(ctx.paths!.kbPath).catch(() => { });
        })
        .catch(() => { });
    }

    const store = useOllamaStore.getState();
    store.initMachineProfile().catch(() => { });
    store.refreshModels().catch(() => { });
    EmbeddingModelService.clearCache();
  },
};

function toInitResult(ctx: InitContext): InitResult {
  if (ctx.status === 'setup') return { status: 'setup' };
  if (ctx.status === 'locked') return { status: 'locked', rootPath: ctx.rootPath };
  if (ctx.status === 'error') return { status: 'error', error: ctx.error };
  return { status: 'ready', rootPath: ctx.rootPath };
}
