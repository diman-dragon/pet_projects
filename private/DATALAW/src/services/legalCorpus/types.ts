export interface LegalActInput {
  jurisdictionCode: string;
  jurisdictionName: string;
  legalSystem?: string | null;
  courtBodyName?: string | null;
  courtBodyLevel?: string | null;
  courtBodyCode?: string | null;
  actType: string;
  title: string;
  actNumber?: string | null;
  adoptedDate?: string | null;
  effectiveFrom?: string | null;
  effectiveTo?: string | null;
  status?: string | null;
  lang?: string | null;
  sourceUrl?: string | null;
  contentRaw: string;
  hash: string;
}

export interface LegalSearchResult {
  chunkId: number;
  actId: number;
  title: string;
  articleRef: string | null;
  text: string;
  rank: number;
  jurisdictionId: number;
}
