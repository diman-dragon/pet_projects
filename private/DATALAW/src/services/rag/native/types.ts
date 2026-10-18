export interface NativeTextChunk {
  text: string;
  chunkIndex: number;
  charStart: number;
  charEnd: number;
  parentIndex?: number;
  articleRef?: string;
}

export interface NativeSearchResult {
  chunkId: number;
  libraryId: number;
  title: string;
  category: string;
  filePath: string;
  chunkText: string;
  chunkIndex: number;
  score: number;
  matchType: 'vector' | 'text' | 'hybrid';
}
