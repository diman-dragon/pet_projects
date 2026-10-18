import type { SystematizationResult } from '../../services/ai/SystematizerAgent';

export interface SystematizerJob {
  id: string;
  fileName: string;
  filePath: string;
  status: 'pending' | 'processing' | 'completed' | 'error' | 'applying' | 'applied';
  result?: SystematizationResult;
  userStatus: 'pending' | 'approved' | 'rejected';
  error?: string;
}

export interface SystematizerStoreState {
  queue: SystematizerJob[];
  isProcessing: boolean;
  isApplying: boolean;
  currentFile: string;
  progress: number;
  addToQueue: (filePaths: string[]) => void;
  startProcessing: () => Promise<void>;
  approveAll: () => void;
  applyAll: () => Promise<void>;
  clearQueue: () => void;
  removeJob: (id: string) => void;
  updateUserStatus: (id: string, status: SystematizerJob['userStatus']) => void;
  updateJobResult: (id: string, result: SystematizationResult) => void;
  updateJobStatus: (id: string, status: SystematizerJob['status'], result?: SystematizationResult, error?: string) => void;
}
