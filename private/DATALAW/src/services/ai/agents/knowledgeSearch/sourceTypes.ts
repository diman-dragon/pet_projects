export interface KnowledgeSearchSource {
  index: number;
  title: string;
  excerpt: string;
  score: number;
  filePath: string;
  charStart?: number;
  charEnd?: number;
  articleRef?: string | null;
}

export interface KnowledgeSearchResponse {
  answer: string;
  sources: KnowledgeSearchSource[];
}
