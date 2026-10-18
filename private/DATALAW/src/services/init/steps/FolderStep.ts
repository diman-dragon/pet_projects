// src/services/init/steps/FolderStep.ts
// Ответственность: идемпотентно создать папочную структуру Dorlane.
// Читает rootPath из контекста, кладёт готовые FolderPaths обратно.

import { join } from '@tauri-apps/api/path';
import { StorageService } from '../../fs/StorageService';
import { FIRM_INIT_SCHEMA } from '../../../constants/schemas/FirmInitSchema';
import { createLogger } from '../../logger/Logger';
import { InitStep, InitContext } from '../types';

const log = createLogger('FolderStep');

export class FolderStep implements InitStep {
  readonly name = 'Folder';

  async run(ctx: InitContext): Promise<InitContext> {
    const { rootPath } = ctx;

    await StorageService.build(rootPath!, FIRM_INIT_SCHEMA.root);

    const kbPath = await join(rootPath!, 'Knowledge_Base');
    const clientsPath = await join(rootPath!, 'Clients');
    const piiVaultPath = await join(rootPath!, 'PII_Vault');
    const legalCorpusPath = await join(rootPath!, 'Legal_Corpus');

    await StorageService.build(kbPath, FIRM_INIT_SCHEMA.knowledge_base);

    log.info('Структура папок готова.');
    return { ...ctx, paths: { rootPath: rootPath!, kbPath, clientsPath, piiVaultPath, legalCorpusPath } };
  }
}
