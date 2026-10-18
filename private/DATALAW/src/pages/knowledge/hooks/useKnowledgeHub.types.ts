import type { SearchResult } from '../../../services/rag/UnifiedSearchService';
import type { LibraryDocumentWithStatus } from '../../../services/rag/knowledgeLibrary/types';

export type KnowledgeTab = 'semantic' | 'fulltext' | 'library' | 'indexing';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  sources?: Array<{
    index: number;
    title: string;
    excerpt: string;
    score: number;
    filePath: string;
    charStart?: number;
    charEnd?: number;
    articleRef?: string | null;
  }>;
}

export interface SemanticState {
  messages: ChatMessage[];
  isStreaming: boolean;
  isThinking: boolean;
  abortController: AbortController | null;
  chatId: string | null;
}

export interface FullTextState {
  query: string;
  results: SearchResult[];
  isSearching: boolean;
  selectedFolder: string | null;
}

export interface LibraryState {
  documents: LibraryDocumentWithStatus[];
  folders: string[];
  selectedFolder: string | null;
  isUploading: boolean;
  isIndexing: boolean;
  reindexProgress: {
    current: number;
    total: number;
    documentName: string;
  } | null;
}
