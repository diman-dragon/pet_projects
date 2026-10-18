// src/services/ai/agents/DocumentClassifierAgent.ts
// Классификатор документов по имени файла + превью текста.
// Шаг 1: детерминированная классификация по имени (без LLM, мгновенно).
// Шаг 2: LLM с первыми 500 символами текста (если шаг 1 не дал уверенного результата).

import { aiApi } from '../../backend/aiApi';
import { getActiveChatModel } from '../config';
import { getContextParams } from '../ModelContextProfile';
import { createLogger } from '../../logger/Logger';
import { AppLanguageService } from '../../config/AppLanguageService';

const log = createLogger('DocumentClassifierAgent');

export type DocumentClass =
  | 'contract'           // Договор / соглашение
  | 'court_decision'     // Решение / определение / постановление суда
  | 'appeal'             // Апелляционная / кассационная жалоба
  | 'complaint'          // Исковое заявление / жалоба
  | 'power_of_attorney'  // Доверенность
  | 'act'                // Акт (приёма-передачи, выполненных работ и т.д.)
  | 'invoice'            // Счёт / счёт-фактура / УПД
  | 'protocol'           // Протокол (судебного заседания, общего собрания и т.д.)
  | 'certificate'        // Свидетельство / справка / выписка
  | 'regulation'         // Нормативный акт / закон / приказ / постановление органа власти
  | 'notice'             // Уведомление / претензия / требование
  | 'application'        // Заявление (не исковое — ходатайство, заявка и т.д.)
  | 'other';             // Иное / не определено

export interface ClassificationResult {
  documentClass: DocumentClass;
  label: string;     // Человекочитаемое название на русском
  confidence: number; // 0–1
  reason: string;    // Краткое объяснение (1 предложение)
}

// ── Детерминированная классификация по имени файла (без LLM) ─────────────────

const NAME_RULES: Array<{ pattern: RegExp; cls: DocumentClass; label: string }> = [
  { pattern: /договор|contract|соглашение|agreement|оферт/i, cls: 'contract', label: 'Договор' },
  { pattern: /решение|определени[ея]|постановление суда|decision/i, cls: 'court_decision', label: 'Судебное решение' },
  { pattern: /апелляц|кассац|надзор|appeal|cassation/i, cls: 'appeal', label: 'Апелляционная жалоба' },
  { pattern: /иск|жалоб[ао]|complaint|claim|заявлени[ея] в суд/i, cls: 'complaint', label: 'Исковое заявление' },
  { pattern: /доверенн|attorney|доверит/i, cls: 'power_of_attorney', label: 'Доверенность' },
  { pattern: /\bакт\b|\bact\b|акт приём|акт выполн/i, cls: 'act', label: 'Акт' },
  { pattern: /счёт[-\s]?фактур|упд|invoice|счет на оплат|счёт на оплат/i, cls: 'invoice', label: 'Счёт-фактура / УПД' },
  { pattern: /\bсчёт\b|\bсчет\b(?![-\s]?фактур)/i, cls: 'invoice', label: 'Счёт' },
  { pattern: /протокол|protocol/i, cls: 'protocol', label: 'Протокол' },
  { pattern: /свидетельств|справк|выписк|certificate|extract/i, cls: 'certificate', label: 'Справка / выписка' },
  { pattern: /закон|фз-\d|приказ №|регламент|regulation|law\b/i, cls: 'regulation', label: 'Нормативный акт' },
  { pattern: /уведомл|претензи|требовани[ея]|notice|demand/i, cls: 'notice', label: 'Уведомление / претензия' },
  { pattern: /ходатайств|заявлени[ея](?! в суд)|application|petition/i, cls: 'application', label: 'Заявление / ходатайство' },
];

function classifyByName(fileName: string): ClassificationResult | null {
  // Убираем расширение и нормализуем
  const name = fileName.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ');
  for (const rule of NAME_RULES) {
    if (rule.pattern.test(name)) {
      return {
        documentClass: rule.cls,
        label: rule.label,
        confidence: 0.80,
        reason: `Определено по имени файла: «${name}»`,
      };
    }
  }
  return null;
}

// ── LLM-промпт (используется только если имя файла не дало результата) ────────

const CLASSIFY_SYSTEM_PROMPT = `Ты — классификатор юридических документов.
Проанализируй имя файла и первые строки текста. Определи тип документа.
Верни СТРОГО JSON без лишнего текста и без markdown:
{
  "documentClass": "contract|court_decision|appeal|complaint|power_of_attorney|act|invoice|protocol|certificate|regulation|notice|application|other",
  "label": "Название типа на русском языке",
  "confidence": число от 0.0 до 1.0,
  "reason": "одно предложение: почему именно этот тип"
}

Типы:
- contract: любой договор, соглашение, оферта
- court_decision: решение, определение, постановление суда (не органа власти)
- appeal: апелляционная, кассационная, надзорная жалоба
- complaint: исковое заявление, жалоба в суд
- power_of_attorney: доверенность
- act: акт приёма-передачи, акт выполненных работ
- invoice: счёт, счёт-фактура, УПД
- protocol: протокол судебного заседания, протокол собрания
- certificate: справка, свидетельство, выписка из реестра
- regulation: федеральный закон, приказ министерства, постановление правительства
- notice: уведомление, претензия, требование об оплате
- application: ходатайство, заявление (не исковое)
- other: не подходит ни к одной категории`;

export const DocumentClassifierAgent = {
  /**
   * Классифицирует документ.
   * Сначала — детерминированная проверка по имени файла (без LLM, ~0 мс).
   * При низкой уверенности — LLM с первыми 500 символами текста.
   */
  async classify(fileName: string, textPreview = ''): Promise<ClassificationResult> {
    // Шаг 1: быстрая классификация по имени
    const byName = classifyByName(fileName);
    if (byName && byName.confidence >= 0.80) return byName;

    // Шаг 2: LLM классификация если имя файла не дало уверенного результата
    const model = await getActiveChatModel();

    const ctxParams = getContextParams('classify');
    const preview = textPreview.slice(0, 500);
    // FIX: добавлена языковая инструкция — без неё LLM возвращал label/reason
    // на языке обучения модели (английский), игнорируя язык UI.
    const langInstruction = await AppLanguageService.getPromptInstruction();

    const messages = [
      { role: 'system' as const, content: `${langInstruction}\n\n${CLASSIFY_SYSTEM_PROMPT}` },
      {
        role: 'user' as const,
        content: `Имя файла: ${fileName}\n\nПервые строки документа:\n${preview || '(текст недоступен)'}`,
      },
    ];

    try {
      const response = await aiApi.chat(model, messages, false, ctxParams);
      const clean = response.content
        .replace(/```json\s*/gi, '')
        .replace(/```\s*/g, '')
        .trim();
      const start = clean.indexOf('{');
      const end = clean.lastIndexOf('}');
      const parsed = JSON.parse(start !== -1 && end > start ? clean.slice(start, end + 1) : clean);

      return {
        documentClass: (parsed.documentClass as DocumentClass) ?? 'other',
        label: parsed.label ?? 'Иное',
        confidence: typeof parsed.confidence === 'number' ? parsed.confidence : 0.50,
        reason: parsed.reason ?? '',
      };
    } catch (err) {
      log.error('DocumentClassifierAgent: LLM parse failed', err);
      return byName ?? {
        documentClass: 'other',
        label: 'Иное',
        confidence: 0.10,
        reason: 'Ошибка классификации',
      };
    }
  },

  /**
   * Синхронная классификация только по имени файла (без LLM).
   * Используется в местах где async нежелателен.
   */
  classifyByNameOnly(fileName: string): ClassificationResult {
    return classifyByName(fileName) ?? {
      documentClass: 'other',
      label: 'Иное',
      confidence: 0.10,
      reason: 'Тип не определён по имени файла',
    };
  },
};
