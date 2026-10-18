import { create } from 'zustand';
import { RagFacade } from '../services/rag/RagFacade';
import { EmbeddingModelService } from '../services/ai/EmbeddingModelService';

export type IndexerStatus = 'idle' | 'checking' | 'running' | 'done' | 'error' | 'no-model';

export interface IndexerProgress {
  currentFile: string;
  filesDone: number;
  filesTotal: number;
  docChunksDone: number;
  docChunksTotal: number;
  chunksDone: number;
  chunksTotal: number;
}

export interface IndexerStats {
  documentsCount: number;
  chunksCount: number;
  embeddingsCount: number;
}

interface IndexerStoreState {
  status: IndexerStatus;
  progress: IndexerProgress | null;
  error: string | null;
  stats: IndexerStats | null;
  startIndexing: (kbPath: string) => Promise<void>;
  startScan: (kbPath: string) => Promise<void>;
  indexSingleFile: (
    filePath: string,
    opts?: { title?: string; category?: string },
  ) => Promise<void>;
  forceReembed: () => Promise<void>;
  abort: () => void;
  reset: () => void;
  loadStats: () => Promise<void>;
  _abortController: AbortController | null;
}

const EMPTY_PROGRESS: IndexerProgress = {
  currentFile: '',
  filesDone: 0,
  filesTotal: 0,
  docChunksDone: 0,
  docChunksTotal: 0,
  chunksDone: 0,
  chunksTotal: 0,
};

const unavailableMessage = (model: string) =>
  `Локальный backend эмбеддингов "${model}" недоступен.`;

export const useIndexerStore = create<IndexerStoreState>((set, get) => ({
  status: 'idle',
  progress: null,
  error: null,
  stats: null,
  _abortController: null,

  loadStats: async () => {
    try {
      const stats = await RagFacade.getStats();
      set({ stats });
    } catch (error) {
      console.warn('[indexerStore] loadStats error:', error);
    }
  },

  abort: () => {
    get()._abortController?.abort();
    set({ status: 'idle', _abortController: null });
  },

  reset: () => {
    get()._abortController?.abort();
    set({ status: 'idle', error: null, progress: null, _abortController: null });
  },

  startIndexing: async (kbPath: string) => {
    get()._abortController?.abort();
    const controller = new AbortController();
    set({
      status: 'checking',
      error: null,
      progress: { ...EMPTY_PROGRESS, currentFile: 'Проверка backend эмбеддингов...' },
      _abortController: controller,
    });

    const model = await EmbeddingModelService.get();
    const modelOk = await RagFacade.isEmbeddingModelAvailable(model);
    if (!modelOk) {
      set({ status: 'no-model', error: unavailableMessage(model) });
      return;
    }

    set({ status: 'running' });
    let totalChunksDone = 0;
    let totalChunksTotal = 0;

    try {
      const result = await RagFacade.indexFolder(kbPath, {
        signal: controller.signal,
        onFileProgress: (fileName: string, fileDone: number, fileTotal: number) => {
          set((state) => ({
            progress: {
              currentFile: fileName,
              filesDone: fileDone,
              filesTotal: fileTotal,
              docChunksDone: 0,
              docChunksTotal: 0,
              chunksDone: state.progress?.chunksDone ?? 0,
              chunksTotal: state.progress?.chunksTotal ?? 0,
            },
          }));
        },
        onChunkProgress: (done: number, total: number) => {
          if (total > 0 && total !== totalChunksTotal) totalChunksTotal += total;
          totalChunksDone = (get().progress?.chunksDone ?? 0) + done;
          set((state) => ({
            progress: {
              ...(state.progress ?? EMPTY_PROGRESS),
              docChunksDone: done,
              docChunksTotal: total,
              chunksDone: totalChunksDone,
              chunksTotal: totalChunksTotal,
            },
          }));
        },
      });

      if (result.indexed === 0 && result.errors.length > 0) {
        set({
          status: 'error',
          error: `Ошибка индексации (${result.errors.length} файлов):\n${result.errors[0]}`,
          _abortController: null,
        });
      } else {
        set({ status: 'done', _abortController: null });
      }
      await get().loadStats();
    } catch (error: any) {
      if (error?.name === 'AbortError') {
        set({ status: 'idle', _abortController: null });
        return;
      }
      set({ status: 'error', error: error?.message ?? String(error), _abortController: null });
    }
  },

  startScan: async (kbPath: string) => {
    get()._abortController?.abort();
    const controller = new AbortController();
    set({
      status: 'running',
      error: null,
      progress: { ...EMPTY_PROGRESS, currentFile: 'Сканирование...' },
      _abortController: controller,
    });

    try {
      await RagFacade.indexFolder(kbPath, {
        signal: controller.signal,
        skipEmbeddings: true,
        onFileProgress: (fileName: string, fileDone: number, fileTotal: number) => {
          set({
            progress: {
              ...EMPTY_PROGRESS,
              currentFile: fileName,
              filesDone: fileDone,
              filesTotal: fileTotal,
            },
          });
        },
      });
      set({ status: 'done', _abortController: null });
      await get().loadStats();
    } catch (error: any) {
      if (error?.name === 'AbortError') {
        set({ status: 'idle', _abortController: null });
        return;
      }
      set({ status: 'error', error: error?.message ?? String(error), _abortController: null });
    }
  },

  indexSingleFile: async (filePath: string, opts?: { title?: string; category?: string }) => {
    const model = await EmbeddingModelService.get();
    const modelOk = await RagFacade.isEmbeddingModelAvailable(model);
    if (!modelOk) {
      set({ status: 'no-model', error: unavailableMessage(model) });
      return;
    }

    set({
      status: 'running',
      error: null,
      progress: {
        ...EMPTY_PROGRESS,
        currentFile: filePath.split(/[\\/]/).pop() ?? filePath,
        filesTotal: 1,
      },
    });

    try {
      await RagFacade.indexFile({
        filePath,
        title: opts?.title,
        category: opts?.category,
        onProgress: (done, total) => {
          set((state) => ({
            progress: {
              ...(state.progress ?? EMPTY_PROGRESS),
              filesDone: 1,
              chunksDone: done,
              chunksTotal: total,
              docChunksDone: done,
              docChunksTotal: total,
            },
          }));
        },
      });
      set({ status: 'done' });
      await get().loadStats();
    } catch (error: any) {
      set({ status: 'error', error: error?.message ?? String(error) });
    }
  },

  forceReembed: async () => {
    get()._abortController?.abort();
    const controller = new AbortController();
    const model = await EmbeddingModelService.get();
    const modelOk = await RagFacade.isEmbeddingModelAvailable(model);
    if (!modelOk) {
      set({ status: 'no-model', error: unavailableMessage(model) });
      return;
    }

    set({
      status: 'running',
      error: null,
      progress: { ...EMPTY_PROGRESS, currentFile: 'Удаление старых эмбеддингов...' },
      _abortController: controller,
    });

    try {
      const result = await RagFacade.forceReembedAll((progress) => {
        set({
          progress: {
            currentFile: progress.currentTitle,
            filesDone: 0,
            filesTotal: 0,
            docChunksDone: progress.currentDocDone,
            docChunksTotal: progress.currentDocTotal,
            chunksDone: progress.done,
            chunksTotal: progress.total,
          },
        });
      }, controller.signal);

      if (result.backendDown) {
        set({
          status: 'error',
          error: `Локальный backend эмбеддингов стал недоступен в процессе переиндексации.\nПроиндексировано: ${result.done} чанков.`,
          _abortController: null,
        });
        return;
      }

      if (result.aborted) {
        set({ status: 'idle', _abortController: null });
        return;
      }

      if (result.done === 0 && result.errors.length > 0) {
        set({
          status: 'error',
          error: `Переиндексация не удалась (${result.errors.length} ошибок).\nПервая ошибка: ${result.errors[0]}`,
          _abortController: null,
        });
      } else {
        set({ status: 'done', _abortController: null });
      }

      await get().loadStats();
    } catch (error: any) {
      if (error?.name === 'AbortError') {
        set({ status: 'idle', _abortController: null });
        return;
      }
      set({ status: 'error', error: error?.message ?? String(error), _abortController: null });
    }
  },
}));
