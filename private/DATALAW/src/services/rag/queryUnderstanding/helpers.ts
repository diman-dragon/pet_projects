import type { PreprocessedQuery, QueryUnderstandingResult } from './types';

export function preprocessQuery(query: string): PreprocessedQuery {
  let normalized = query.toLowerCase().trim().replace(/\s+/g, ' ');

  const corrections: Record<string, string> = {
    'гкрф': 'гк рф',
    'укрф': 'ук рф',
    'нкрф': 'нк рф',
    'ткрф': 'тк рф',
    'кодексрф': 'кодекс рф',
    'стт': 'ст',
    'статьяст': 'статья',
    'штрафы': 'штраф',
    'договора': 'договор',
  };

  for (const [wrong, right] of Object.entries(corrections)) {
    normalized = normalized.replace(new RegExp(wrong, 'g'), right);
  }

  return {
    normalized,
    intentHint: classifyIntentSimple(normalized),
  };
}

export function classifyIntentSimple(query: string): string {
  const conversationalPatterns = [
    /^(привет|здравствуй|здрасте|добрый\s+\w+|хай|hello|hi\b)/i,
    /\b(ты лайла|ты laila|кто ты|кто ты такая|как тебя зовут|твоё имя|твое имя|ты ии|ты бот|ты робот|ты программа|ты ассистент)\b/i,
    /\b(что ты умеешь|что ты делаешь|чем ты занимаешься|расскажи о себе|представься)\b/i,
    // FIX: эмоциональные паттерны вынесены в отдельный intent "frustration"
    // чтобы не блокировать RAG-поиск когда пользователь раздражён но спрашивает по делу.
    // Пример: "что за хуйня с этим договором — объясни статью 450"
    // НЕ попадает в conversational и продолжает нормальный pipeline.
    /^[а-яёa-z\s]{1,6}[!?]+$/i,
  ];
  if (conversationalPatterns.some((pattern) => pattern.test(query))) return 'conversational';

  // FIX: frustration — эмоция, но не разговор. Продолжаем RAG.
  // Определяется отдельно; в fastUnderstand/intentMap маппится в 'factual'
  // чтобы pipeline не прерывался на handleConversationalQuery.
  const frustrationPatterns = [
    /\b(срань|хрень|хуйня|хуета|еRunда|чушь|бред|тупость|дебилизм|что за\s+\w+ть)\b/i,
  ];
  if (frustrationPatterns.some((pattern) => pattern.test(query))) return 'frustration';

  if (/\b(кто такой|кто такая|биография|резюме|партнер|юрист|адвокат|судья)\b/.test(query)) return 'person_lookup';
  if (/\b(статья|ст\.?|кодекс|закон|гк|ук|нк|тк|коап|фз)\b/.test(query)) return 'legal_doc';
  if (/\b(как|порядок|процедура|пошагово|инструкция|подать|оформить)\b/.test(query)) return 'procedure';
  return 'generic';
}

export function fastUnderstand(query: string, intentHint: string): QueryUnderstandingResult {
  const leadingFillers = /^\s*(расскажи(?:те)?|объясни(?:те)?|поясни|напомни|покажи|скажи|дай(?:те)?(?:\s+мне)?|найди|что такое|что значит)\s+/i;
  const rewritten = query.trim().replace(leadingFillers, '').trim() || query.trim();

  const intentMap: Record<string, QueryUnderstandingResult['intent']> = {
    conversational: 'conversational',
    // FIX: frustration → factual, не прерываем pipeline
    frustration: 'factual',
    person_lookup: 'biographical',
    legal_doc: 'legal',
    procedure: 'procedural',
    generic: 'factual',
  };

  return {
    rewritten_query: rewritten,
    expansions: generateNameVariants(rewritten),
    intent: intentMap[intentHint] ?? 'factual',
    lang: 'ru',
  };
}

export function generateNameVariants(query: string): string[] {
  const variants = new Set<string>();

  const ruGenRegex = /([\p{L}]{3,})(ого|его|ому|ему|ской)\b/gu;
  let match: RegExpExecArray | null;
  while ((match = ruGenRegex.exec(query)) !== null) {
    const stem = match[1];
    const base = query.replace(match[0], stem);
    variants.add(base + 'а');
    variants.add(base + 'ська');
    variants.add(base + 'ський');
    variants.add(base);
  }

  const ukrRegex = /([\p{L}]{3,})(ська|ський|зька|зький)\b/gu;
  while ((match = ukrRegex.exec(query)) !== null) {
    const stem = match[1];
    const base = query.replace(match[0], stem);
    variants.add(base + 'ского');
    variants.add(base + 'ский');
    variants.add(base);
  }

  variants.delete(query);
  return Array.from(variants).slice(0, 3);
}

export function extractJson(content: string): Record<string, any> {
  try {
    return JSON.parse(content);
  } catch {}

  const markdownMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (markdownMatch) {
    try {
      return JSON.parse(markdownMatch[1]);
    } catch {}
  }

  const objectMatch = content.match(/\{[\s\S]*"rewritten_query"[\s\S]*\}/);
  if (objectMatch) {
    try {
      return JSON.parse(objectMatch[0]);
    } catch {}
  }

  throw new Error('No valid JSON found');
}
