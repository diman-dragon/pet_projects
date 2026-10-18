import type { SearchResult } from '../../../../services/rag/UnifiedSearchService';

export interface FullTextSearchTabProps {
  query: string;
  results: SearchResult[];
  isSearching: boolean;
  folders: string[];
  selectedFolder: string | null;
  onSearch: (query: string, folder: string | null) => void;
}

export interface FilterState {
  titleQuery: string;
  category: string;
  language: string;
  folder: string;
  fileType: string;
  matchType: 'all' | 'text' | 'vector' | 'hybrid';
  minScore: number;
  topK: number;
}
