export interface SearchResult {
  chunkId: number;
  libraryId: number;
  title: string;
  category: string;
  filePath: string;
  chunkText: string;
  chunkIndex: number;
  score: number;
  matchType: 'vector' | 'text' | 'hybrid';
  lang?: string;
  charStart?: number;
  charEnd?: number;
  articleRef?: string | null;
  parentId?: number;
  parentText?: string;
}
export interface UnifiedSearchOptions {
  query: string;
  queryVector?: Float32Array;
  topK?: number;
  category?: string;
  hybrid?: boolean;
  useReranker?: boolean;
  threshold?: number;
  rewrittenQuery?: string;
  expansions?: string[];
  intent?: string;
  hydeSnippet?: string;
  onDebug?: (trace: any) => void;
}
