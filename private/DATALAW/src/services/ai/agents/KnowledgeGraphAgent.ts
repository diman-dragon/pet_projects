import { aiApi } from '../../backend/aiApi';
import { createLogger } from '../../logger/Logger';
import { getContextParams } from '../ModelContextProfile';
import { getActiveChatModel } from '../config';
import { KNOWLEDGE_GRAPH_SYSTEM_PROMPT } from '../prompts/knowledgeGraph.prompt';

const log = createLogger('KnowledgeGraphAgent');

export type KgEntityType = 'PERSON' | 'ORGANIZATION' | 'OTHER';

export type KgRelationType =
  | 'PARTY_TO'
  | 'REPRESENTS'
  | 'FILED_BY'
  | 'DECIDED_BY'
  | 'COUNTERPARTY'
  | 'EMPLOYED_BY'
  | 'INVOLVED_IN'
  | 'OTHER';

export interface KgEntity {
  name: string;
  entity_type: KgEntityType;
  canonical_name: string;
  description?: string | null;
  confidence: number;
}

export interface KgRelation {
  source: string;
  target: string;
  relation_type: KgRelationType;
  label?: string | null;
  weight: number;
  confidence: number;
}

export interface KgExtractionResult {
  entities: KgEntity[];
  relations: KgRelation[];
  usedLlm: boolean;
}

const DETERMINISTIC_RULES: Array<{
  pattern: RegExp;
  entity_type: KgEntityType;
  transform?: (raw: string) => string;
}> = [
  {
    pattern: /ИНН\s*:?\s*(\d{10}|\d{12})/gi,
    entity_type: 'ORGANIZATION',
    transform: (raw) => `ИНН ${raw.replace(/ИНН\s*:?\s*/i, '').trim()}`,
  },
];

const VALID_ENTITY_TYPES = new Set<KgEntityType>(['PERSON', 'ORGANIZATION', 'OTHER']);
const VALID_RELATION_TYPES = new Set<KgRelationType>([
  'PARTY_TO',
  'REPRESENTS',
  'FILED_BY',
  'DECIDED_BY',
  'COUNTERPARTY',
  'EMPLOYED_BY',
  'INVOLVED_IN',
  'OTHER',
]);

const ACTOR_TYPES = new Set<KgEntityType>(['PERSON', 'ORGANIZATION']);

function canon(value: string): string {
  return String(value).toLowerCase().replace(/\s+/g, ' ').trim();
}

function parseEntityType(value: unknown): KgEntityType {
  return VALID_ENTITY_TYPES.has(value as KgEntityType) ? (value as KgEntityType) : 'OTHER';
}

function parseRelationType(value: unknown): KgRelationType {
  return VALID_RELATION_TYPES.has(value as KgRelationType)
    ? (value as KgRelationType)
    : 'OTHER';
}

function parseConfidence(value: unknown, fallback = 0.7): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.min(1, Math.max(0, parsed)) : fallback;
}

function extractDeterministic(text: string): KgEntity[] {
  const found = new Map<string, KgEntity>();

  for (const rule of DETERMINISTIC_RULES) {
    for (const match of text.matchAll(rule.pattern)) {
      const raw = match[0].trim();
      const name = rule.transform ? rule.transform(raw) : raw;
      const canonical = canon(name);
      if (!canonical || found.has(canonical)) continue;

      found.set(canonical, {
        name,
        entity_type: rule.entity_type,
        canonical_name: canonical,
        confidence: 0.9,
      });
    }
  }

  return [...found.values()];
}

function parseJsonResponse(raw: string): { entities: KgEntity[]; relations: KgRelation[] } {
  const clean = raw.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim();
  const start = clean.indexOf('{');
  const end = clean.lastIndexOf('}');
  if (start === -1 || end <= start) {
    throw new Error('No JSON in LLM response');
  }

  const parsed = JSON.parse(clean.slice(start, end + 1));

  const entities: KgEntity[] = (Array.isArray(parsed.entities) ? parsed.entities : [])
    .filter((entry: any) => entry?.name && String(entry.name).trim().length > 1)
    .map((entry: any): KgEntity => ({
      name: String(entry.name).trim(),
      entity_type: parseEntityType(entry.entity_type),
      canonical_name: canon(entry.canonical_name ?? entry.name),
      description: entry.description ? String(entry.description).slice(0, 300) : null,
      confidence: parseConfidence(entry.confidence),
    }))
    .filter(
      (entity: KgEntity) =>
        entity.confidence >= 0.5 && ACTOR_TYPES.has(entity.entity_type) && entity.canonical_name,
    )
    .slice(0, 30);

  const actorCanonicals = new Set(entities.map((entity) => entity.canonical_name));

  const relations: KgRelation[] = (Array.isArray(parsed.relations) ? parsed.relations : [])
    .filter((entry: any) => {
      const source = canon(entry?.source ?? '');
      const target = canon(entry?.target ?? '');
      return source && target && source !== target && actorCanonicals.has(source) && actorCanonicals.has(target);
    })
    .map((entry: any): KgRelation => ({
      source: canon(entry.source),
      target: canon(entry.target),
      relation_type: parseRelationType(entry.relation_type),
      label: entry.label ? String(entry.label).slice(0, 100) : null,
      weight: parseConfidence(entry.weight, 1),
      confidence: parseConfidence(entry.confidence),
    }))
    .slice(0, 50);

  return { entities, relations };
}

export const KnowledgeGraphAgent = {
  async extract(text: string, maxTextLen = 2500): Promise<KgExtractionResult> {
    const deterministicEntities = extractDeterministic(text);
    const minTextLength = 80;

    log.info('[KG] Deterministic actor extraction complete', {
      textLength: text.length,
      entityCount: deterministicEntities.length,
    });

    if (text.length < minTextLength) {
      return { entities: deterministicEntities, relations: [], usedLlm: false };
    }

    try {
      const model = await getActiveChatModel();
      const ctxParams = getContextParams('kg_extract');
      const preview = text.slice(0, maxTextLen);

      const response = await aiApi.chat(
        model,
        [
          { role: 'system', content: KNOWLEDGE_GRAPH_SYSTEM_PROMPT },
          { role: 'user', content: `Извлеки участников и связи между ними:\n\n${preview}` },
        ],
        false,
        ctxParams,
      );

      if (!response?.content) {
        log.warn('[KG] LLM returned empty actor extraction response');
        return { entities: deterministicEntities, relations: [], usedLlm: true };
      }

      const { entities: llmEntities, relations } = parseJsonResponse(response.content);
      const entityMap = new Map<string, KgEntity>();

      for (const entity of [...deterministicEntities, ...llmEntities]) {
        if (!ACTOR_TYPES.has(entity.entity_type)) continue;
        if (!entityMap.has(entity.canonical_name)) {
          entityMap.set(entity.canonical_name, entity);
        }
      }

      return { entities: [...entityMap.values()], relations, usedLlm: true };
    } catch (error) {
      log.error('[KG] Actor extraction failed', error);
      return { entities: deterministicEntities, relations: [], usedLlm: false };
    }
  },

  extractSync(text: string): Pick<KgExtractionResult, 'entities' | 'usedLlm'> {
    return { entities: extractDeterministic(text), usedLlm: false };
  },
};
