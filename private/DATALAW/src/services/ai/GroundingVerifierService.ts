import { ConnectionManager } from '../db/ConnectionManager';

export interface GroundingIssue {
  citation: string;
  reason: string;
}

export interface GroundingReport {
  ok: boolean;
  citations: string[];
  issues: GroundingIssue[];
}

const CITATION_RE = /\[(\d+):([^\]]+)\]/g;

export const GroundingVerifierService = {
  async verifyDraft(body: string): Promise<GroundingReport> {
    const db = ConnectionManager.get('legal_corpus');
    const citations = [...body.matchAll(CITATION_RE)].map((match) => ({
      raw: match[0],
      chunkId: Number(match[1]),
      articleRef: match[2].trim(),
    }));

    const issues: GroundingIssue[] = [];
    if (citations.length === 0) {
      issues.push({ citation: 'none', reason: 'Draft has no legal corpus citations' });
    }

    for (const citation of citations) {
      const [chunk] = await db.select<{ id: number; article_ref: string | null; text: string }[]>(
        `SELECT id, article_ref, text FROM act_chunks WHERE id = $1`,
        [citation.chunkId],
      );
      if (!chunk) {
        issues.push({ citation: citation.raw, reason: 'Referenced chunk does not exist' });
        continue;
      }
      if (citation.articleRef && chunk.article_ref && citation.articleRef !== chunk.article_ref) {
        issues.push({ citation: citation.raw, reason: 'Article reference does not match corpus chunk' });
      }
    }

    return {
      ok: issues.length === 0,
      citations: citations.map((citation) => citation.raw),
      issues,
    };
  },
};
