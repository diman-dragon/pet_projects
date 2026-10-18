// src/pages/knowledge/types/index.ts

export type Jurisdiction = 'all' | 'ЕСПЧ' | 'РФ' | 'USA' | 'UK';

export type DocStatus = 'all' | 'active' | 'complaint' | 'decided';

export interface DocSection {
  id: string;
  label: string;
}

export interface CaseDocument {
  id: string;
  title: string;
  caseNumber: string;
  court: string;
  jurisdiction: Exclude<Jurisdiction, 'all'>;
  year: number;
  status: Exclude<DocStatus, 'all'>;
  snippet: string;
  url?: string;
  tags: string[];
  sections: DocSection[];
  body: string;
  vector?: number[];
}
