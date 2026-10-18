export type PipelineStatus = 'running' | 'setup' | 'locked' | 'error';

export interface FolderPaths {
  rootPath: string;
  kbPath: string;
  clientsPath: string;
  piiVaultPath: string;
  legalCorpusPath: string;
}

export interface InitContext {
  status: PipelineStatus;
  rootPath?: string;
  paths?: FolderPaths;
  error?: string;
  masterKeyReady?: boolean;
  masterPassword?: string;
}

export interface InitStep {
  readonly name: string;
  run(ctx: InitContext): Promise<InitContext>;
}

export type InitStatus = 'ready' | 'setup' | 'locked' | 'error' | 'license';

export interface InitResult {
  status: InitStatus;
  rootPath?: string;
  error?: string;
}
