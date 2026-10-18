import { aiApi } from '../../backend/aiApi';
import { ConnectionManager } from '../../db/ConnectionManager';
import { LegalSearchService } from '../../legalCorpus/LegalSearchService';
import { CaseAccessService } from '../../security/CaseAccessService';
import type { UserProfile } from '../../../types/entities';
import { getActiveChatModel } from '../config';
import { getContextParams } from '../ModelContextProfile';
import { GroundingVerifierService } from '../GroundingVerifierService';
import { DRAFT_SYSTEM_PROMPT } from '../prompts';
import type { StrategyStep } from './DefenseStrategyAgent';

function extractJsonObject(text: string): { title?: string; body?: string } {
  const clean = text.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim();
  const start = clean.indexOf('{');
  const end = clean.lastIndexOf('}');
  return JSON.parse(start !== -1 && end > start ? clean.slice(start, end + 1) : clean);
}

export const DocumentDraftAgent = {
  async draft(caseId: number, actor: UserProfile, strategyStep: StrategyStep): Promise<{ draftId: number; title: string; body: string }> {
    await CaseAccessService.requireCaseAccess(actor, caseId, 'document:draft');

    const clientsDb = ConnectionManager.get('clients_main');
    const legalChunkIds = strategyStep.legal_basis.map((basis) => basis.chunk_id);
    const factIds = strategyStep.supporting_facts.map((fact) => fact.case_document_id);
    const legalChunks = await LegalSearchService.getChunksByIds(legalChunkIds);

    const facts = [];
    for (const id of factIds) {
      const [fact] = await clientsDb.select<{ id: number; file_name: string; summary: string | null }[]>(
        `SELECT id, file_name, summary FROM case_documents WHERE id = $1 AND case_id = $2`,
        [id, caseId],
      );
      if (fact) facts.push(fact);
    }

    const [caseRow] = await clientsDb.select<{ jurisdiction_id: number | null; case_type: string | null }[]>(
      `SELECT jurisdiction_id, case_type FROM cases WHERE id = $1`,
      [caseId],
    );

    let template = null;
    if (caseRow?.jurisdiction_id) {
      const [row] = await ConnectionManager.get('legal_corpus').select<{ title: string; body: string }[]>(
        `SELECT title, body FROM document_templates
         WHERE (jurisdiction_id = $1 OR jurisdiction_id IS NULL)
           AND (case_type = $2 OR case_type IS NULL)
         ORDER BY jurisdiction_id DESC, case_type DESC
         LIMIT 1`,
        [caseRow.jurisdiction_id, caseRow.case_type ?? null],
      );
      template = row;
    }

    const model = await getActiveChatModel();
    const response = await aiApi.chat(
      model,
      [
        { role: 'system', content: DRAFT_SYSTEM_PROMPT },
        {
          role: 'user',
          content: JSON.stringify({ strategyStep, legalChunks, facts, template }, null, 2),
        },
      ],
      false,
      getContextParams('analyze'),
    );

    const parsed = extractJsonObject(response.content);
    const title = parsed.title ?? 'Draft document';
    const body = parsed.body ?? '';
    const report = await GroundingVerifierService.verifyDraft(body);
    const groundingStatus = report.ok ? 'PASSED' : 'FAILED';

    await clientsDb.execute(
      `INSERT INTO document_drafts (case_id, title, body, grounding_status, verifier_report)
       VALUES ($1, $2, $3, $4, $5)`,
      [caseId, title, body, groundingStatus, JSON.stringify(report)],
    );
    const [draft] = await clientsDb.select<{ id: number }[]>(
      `SELECT id FROM document_drafts WHERE case_id = $1 ORDER BY id DESC LIMIT 1`,
      [caseId],
    );
    if (!draft) throw new Error('Draft was not created');

    return { draftId: draft.id, title, body };
  },
};
