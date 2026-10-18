export interface KnowledgeSourceSelection {
  index: number;
  title: string;
  excerpt: string;
  filePath: string;
  charStart?: number;
  charEnd?: number;
  articleRef?: string | null;
}
