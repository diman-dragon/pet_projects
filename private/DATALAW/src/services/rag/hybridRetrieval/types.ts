export interface RetrievalCandidate {
  chunkId: number;
  libraryId: number;
  title: string;
  category: string;
  filePath: string;
  chunkText: string;
  chunkIndex: number;
  charStart?: number;
  charEnd?: number;
  articleRef?: string | null;
  parentId?: number;
  parentText?: string;
  vectorScore?: number;
  ftsScore?: number;
  metadataScore?: number;
  sourceImportance?: number;
  rrfScore: number;
  fastRerankScore?: number;
  llmRerankScore?: number;
  matchType: 'vector' | 'fts' | 'hybrid';
}

export interface RetrievalTrace {
  plannedQueries: Array<{ query: string; kind: string; weight: number }>;
  vectorResults: Array<{ query: string; kind: string; results: TraceResult[] }>;
  ftsResults: Array<{ query: string; kind: string; results: TraceResult[] }>;
  mergedResults: Array<{
    chunkId: number;
    title: string;
    category: string;
    filePath: string;
    rrfScore: number;
    matchType: RetrievalCandidate['matchType'];
    vectorScore?: number;
    ftsScore?: number;
  }>;
}
export interface HybridRetrievalOptions {
  query: string;
  rewrittenQuery?: string;
  expansions?: string[];
  intent?: string;
  hydeSnippet?: string;
  queryLang?: string;
  topK?: number;
  category?: string;
  onDebug?: (trace: RetrievalTrace) => void;
  queryVector?: Float32Array;
}

export interface RetrievalHit {
  chunkId: number;
  score: number;
  data: any;
}

export interface SimpleQueryResults {
  query: string;
  vectorResults: RetrievalHit[];
  ftsResults: RetrievalHit[];
}

export interface TraceResult {
  chunkId: number;
  title: string;
  score: number;
  category: string;
  filePath: string;
}
