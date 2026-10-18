import { aiApi } from '../../backend/aiApi';
import { ConnectionManager } from '../../db/ConnectionManager';
import { LegalSearchService } from '../../legalCorpus/LegalSearchService';
import { CaseAccessService } from '../../security/CaseAccessService';
import type { UserProfile } from '../../../types/entities';
import { getActiveChatModel } from '../config';
import { getContextParams } from '../ModelContextProfile';
import { STRATEGY_SYSTEM_PROMPT } from '../prompts';

export interface StrategyStep {
  claim: string;
  legal_basis: Array<{ act_id: number; article_ref: string; chunk_id: number }>;
  supporting_facts: Array<{ case_document_id: number }>;
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
}

function extractJsonObject(text: string): unknown {
  const clean = text.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim();
  const start = clean.indexOf('{');
  const end = clean.lastIndexOf('}');
  return JSON.parse(start !== -1 && end > start ? clean.slice(start, end + 1) : clean);
}

export const DefenseStrategyAgent = {
  async build(caseId: number, actor: UserProfile): Promise<StrategyStep[]> {
    await CaseAccessService.requireCaseAccess(actor, caseId, 'case:strategy');

    const db = ConnectionManager.get('clients_main');
    const [caseRow] = await db.select<{
      id: number;
      title: string;
      summary: string | null;
      violation_type: string | null;
      authority_involved: string | null;
      jurisdiction_id: number | null;
    }[]>(
      `SELECT id, title, summary, violation_type, authority_involved, jurisdiction_id
       FROM cases WHERE id = $1`,
      [caseId],
    );
    if (!caseRow) throw new Error('Case not found');

    const documents = await db.select<{
      id: number;
      category: string;
      file_name: string;
      summary: string | null;
    }[]>(
      `SELECT id, category, file_name, summary
       FROM case_documents
       WHERE case_id = $1 AND deleted_at IS NULL
       ORDER BY created_at DESC
       LIMIT 20`,
      [caseId],
    );

    const query = [caseRow.violation_type, caseRow.authority_involved, caseRow.summary, caseRow.title]
      .filter(Boolean)
      .join(' ');
    const legal = await LegalSearchService.searchText(query || caseRow.title, {
      jurisdictionId: caseRow.jurisdiction_id,
      limit: 8,
    });

    const model = await getActiveChatModel();
    const response = await aiApi.chat(
      model,
      [
        { role: 'system', content: STRATEGY_SYSTEM_PROMPT },
        {
          role: 'user',
          content: JSON.stringify({ case: caseRow, documents, legal_chunks: legal }, null, 2),
        },
      ],
      false,
      getContextParams('analyze'),
    );

    const parsed = extractJsonObject(response.content) as { steps?: StrategyStep[] };
    const steps = Array.isArray(parsed.steps) ? parsed.steps : [];

    for (const step of steps) {
      await db.execute(
        `INSERT INTO defense_strategy_steps (
          case_id, claim, legal_basis_json, supporting_facts_json, confidence, created_by_agent
        ) VALUES ($1, $2, $3, $4, $5, $6)`,
        [
          caseId,
          step.claim,
          JSON.stringify(step.legal_basis ?? []),
          JSON.stringify(step.supporting_facts ?? []),
          step.confidence ?? 'LOW',
          'DefenseStrategyAgent',
        ],
      );
    }

    return steps;
  },
};
