import { create } from 'zustand';
import { createScanActions } from './systematizer/scanActions';
import { createEditActions } from './systematizer/editActions';
import type { SystematizerJob, SystematizerStoreState } from './systematizer/types';

export type { SystematizerJob } from './systematizer/types';

export const useSystematizerStore = create<SystematizerStoreState>((set, get) => ({
  queue: [] as SystematizerJob[],
  isProcessing: false,
  isApplying: false,
  currentFile: '',
  progress: 0,
  ...createScanActions(set, get),
  ...createEditActions(set, get),
}));
