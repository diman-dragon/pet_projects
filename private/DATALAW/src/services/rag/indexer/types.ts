export interface IndexOptions {
  filePath: string;
  title?: string;
  category?: string;
  metadata?: Record<string, unknown>;
  onProgress?: (done: number, total: number) => void;
  signal?: AbortSignal;
  skipEmbeddings?: boolean;
}

export interface IndexResult {
  libraryId: number;
  chunkCount: number;
  skipped: boolean;
  warning?: string;
}

export interface IndexFolderOptions {
  onFileProgress?: (name: string, index: number, total: number) => void;
  onChunkProgress?: (done: number, total: number) => void;
  signal?: AbortSignal;
  skipEmbeddings?: boolean;
}
