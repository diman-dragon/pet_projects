import type { LibraryState } from '../useKnowledgeHub.types';
import { DEFAULT_FOLDERS } from './constants';

export function createInitialLibraryState(): LibraryState {
  return {
    documents: [],
    folders: DEFAULT_FOLDERS,
    selectedFolder: null,
    isUploading: false,
    isIndexing: false,
    reindexProgress: null,
  };
}
