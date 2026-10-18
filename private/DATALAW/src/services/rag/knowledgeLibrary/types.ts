export interface LibraryDocument {
  id: number;
  title: string;
  file_path: string;
  file_type: string;
  category: string;
  hash: string | null;
  metadata: string | null;
  indexed_at: string | null;
  created_at: string;
  updated_at: string;
  chunk_count?: number;
  embedding_count?: number;
}

export interface LibraryDocumentWithStatus extends LibraryDocument {
  isIndexed: boolean;
  indexJobStatus?: 'pending' | 'running' | 'done' | 'error' | null;
}

export interface LibraryIndexJobStatus {
  status: string;
  totalChunks: number;
  doneChunks: number;
  errorMsg: string | null;
}
