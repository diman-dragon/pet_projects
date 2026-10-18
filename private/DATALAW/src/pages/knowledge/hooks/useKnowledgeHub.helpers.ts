import type { Dispatch, SetStateAction } from 'react';
import type { LibraryState, FullTextState, SemanticState, KnowledgeTab } from './useKnowledgeHub.types';

export interface KnowledgeHubViewModel {
  activeTab: KnowledgeTab;
  setActiveTab: Dispatch<SetStateAction<KnowledgeTab>>;
  semanticState: SemanticState;
  sendMessage: (query: string) => void;
  abortSemantic: () => void;
  clearChat: () => void;
  fullTextState: FullTextState;
  searchFullText: (query: string, folder?: string | null) => Promise<void>;
  libraryState: LibraryState;
  setLibraryState: Dispatch<SetStateAction<LibraryState>>;
  uploadDocuments: (files?: string[]) => Promise<void>;
  loadDocuments: () => Promise<void>;
  moveToFolder: (documentId: number, folderPath: string) => Promise<void>;
  reindexDocument: (documentId: number) => Promise<void>;
  reindexAll: () => Promise<void>;
  deleteDocument: (documentId: number) => Promise<void>;
}
