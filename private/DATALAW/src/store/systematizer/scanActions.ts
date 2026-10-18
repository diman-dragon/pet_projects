import { SystematizerAgent } from '../../services/ai/SystematizerAgent';
import { createLogger } from '../../services/logger/Logger';
import type { SystematizerJob, SystematizerStoreState } from './types';

const log = createLogger('SystematizerScanActions');

export function createScanActions(set: any, get: () => SystematizerStoreState) {
  return {
    addToQueue: (filePaths: string[]) => {
      const jobs: SystematizerJob[] = filePaths.map((filePath) => ({
        id: Math.random().toString(36).slice(2),
        fileName: filePath.split(/[\\/]/).pop() || filePath,
        filePath,
        status: 'pending',
        userStatus: 'pending',
      }));
      set((state: SystematizerStoreState) => ({ queue: [...state.queue, ...jobs] }));
      if (!get().isProcessing && !get().isApplying) void get().startProcessing();
    },
    startProcessing: async () => {
      if (get().isProcessing) return;
      set({ isProcessing: true });
      while (true) {
        const job = get().queue.find((item) => item.status === 'pending');
        if (!job) break;
        set({ currentFile: job.fileName });
        get().updateJobStatus(job.id, 'processing');
        try {
          const result = await SystematizerAgent.analyzeDocument(job.fileName, job.filePath);
          get().updateJobStatus(job.id, result ? 'completed' : 'error', result ?? undefined, result ? undefined : 'AI analysis failed');
        } catch (err: any) {
          log.error('Job failed', { job, err });
          get().updateJobStatus(job.id, 'error', undefined, err?.message || 'Unknown error');
        }
      }
      set({ isProcessing: false, currentFile: '' });
    },
    clearQueue: () => set({ queue: [], currentFile: '', progress: 0, isProcessing: false, isApplying: false }),
    removeJob: (id: string) => set((state: SystematizerStoreState) => ({ queue: state.queue.filter((job) => job.id !== id) })),
  };
}
