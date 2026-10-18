// src/services/ai/agents/RiskScannerAgent.ts
// Сканер рисков: пропущенные сроки, дела без сводки

import { aiApi } from '../../backend/aiApi';
import { getActiveChatModel } from '../config';
import { AppLanguageService } from '../../config/AppLanguageService';
import { getContextParams } from '../ModelContextProfile';
import { createLogger } from '../../logger/Logger';

const log = createLogger('RiskScannerAgent');

export type RiskSeverity = 'critical' | 'high' | 'medium' | 'low';

export interface DetectedRisk {
  type: 'missed_deadline' | 'no_summary' | 'overdue_task' | 'inconsistency' | 'other';
  severity: RiskSeverity;
  description: string;
  entityId?: number;
  entityName?: string;
  suggestedAction?: string;
}

export interface RiskScanResult {
  risks: DetectedRisk[];
  scannedAt: string;
  confidence: number;
}

export interface RiskScanInput {
  cases?: Array<{
    id: number;
    name: string;
    deadline?: string | null;
    summary?: string | null;
    status?: string;
    mandateCheck?: string | null;
    riskLevel?: string | null;
    violationType?: string | null;
    authorityInvolved?: string | null;
  }>;
  appeals?: Array<{
    id: number;
    name: string;
    deadline?: string | null;
    summary?: string | null;
  }>;
  today?: string; // ISO date, defaults to now
}

const RISK_SYSTEM_PROMPT = `Ты — аналитик рисков юридической платформы Dorlane.
Проанализируй данные по делам и апелляциям. Выяви реальные риски на основе дат и статусов.

Верни СТРОГО JSON без лишнего текста, markdown-блоков или пояснений:
{
  "risks": [
    {
      "type": "missed_deadline | no_summary | overdue_task | inconsistency | other",
      "severity": "critical | high | medium | low",
      "description": "конкретное описание риска с датами и именами",
      "entityId": число или null,
      "entityName": "название дела/апелляции или null",
      "suggestedAction": "конкретное рекомендуемое действие"
    }
  ],
  "confidence": число от 0.0 до 1.0
}

Правила severity:
- critical: срок истёк ИЛИ истекает сегодня/завтра (diffDays <= 1)
- high: срок через 2–7 дней
- medium: срок через 8–30 дней ИЛИ дело без AI-сводки (no_summary)
- low: прочие несоответствия

Правила type:
- missed_deadline: дело не закрыто, а срок уже прошёл
- no_summary: дело существует но AI-сводка отсутствует — риск потери контекста
- overdue_task: задача просрочена без закрытия
- inconsistency: противоречивые данные (например, статус «закрыто» но срок в будущем)

Важно: возвращай ТОЛЬКО реальные риски. Не добавляй риски «для полноты».
Если данных нет или все дела в порядке — верни пустой массив risks.`;

const HUMAN_RIGHTS_RISK_RULES = `
Human-rights workflow rules:
- Do not include PII in prompts or results. Refer to people only by display_code if present.
- Flag mandate_not_verified when mandate_check is PENDING.
- Flag out_of_mandate when mandate_check is REJECTED_OUT_OF_MANDATE.
- Flag critical_subject_risk when risk_level is CRITICAL.
- Flag authority_involved when authority_involved is present and the case is not closed.
- AI output is suggested only; final risk fields must be confirmed by a human and audited.
`;

export const RiskScannerAgent = {
  async scan(input: RiskScanInput): Promise<RiskScanResult> {
    const model = await getActiveChatModel();
    if (!model) throw new Error('RiskScannerAgent: модель не выбрана');

    // Сначала делаем быструю локальную проверку
    const localRisks = RiskScannerAgent.scanLocal(input);

    // Если данных мало — возвращаем локальный результат без LLM
    const totalEntities = (input.cases?.length ?? 0) + (input.appeals?.length ?? 0);
    if (totalEntities === 0) {
      return { risks: [], scannedAt: new Date().toISOString(), confidence: 1 };
    }

    const today = input.today ?? new Date().toISOString().split('T')[0];
    const langInstruction = await AppLanguageService.getPromptInstruction();
    const ctxParams = getContextParams('risk');

    const dataText = JSON.stringify({ today, cases: input.cases ?? [], appeals: input.appeals ?? [] }, null, 2);

    const messages = [
      { role: 'system' as const, content: `${langInstruction}\n\n${RISK_SYSTEM_PROMPT}\n\n${HUMAN_RIGHTS_RISK_RULES}` },
      { role: 'user' as const, content: `Данные для анализа:\n\`\`\`json\n${dataText}\n\`\`\`` },
    ];

    try {
      const response = await aiApi.chat(model, messages, false, ctxParams);
      const fullText = response.content;

      const clean = fullText.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim();
      const start = clean.indexOf('{');
      const end = clean.lastIndexOf('}');
      const parsed = JSON.parse(start !== -1 && end > start ? clean.slice(start, end + 1) : clean);

      // Мердж локальных рисков с LLM-рисками, дедупликация по entityId+type
      const llmRisks: DetectedRisk[] = Array.isArray(parsed.risks) ? parsed.risks : [];
      const merged = mergeRisks(localRisks, llmRisks);

      return {
        risks: merged,
        scannedAt: new Date().toISOString(),
        confidence: typeof parsed.confidence === 'number' ? parsed.confidence : 0.7,
      };
    } catch (err) {
      log.error('RiskScannerAgent: LLM parse failed, using local results', err);
      // Fallback на локальные результаты
      return { risks: localRisks, scannedAt: new Date().toISOString(), confidence: 0.6 };
    }
  },

  /**
   * Быстрая детерминированная проверка без LLM.
   * Используется как основа и как fallback.
   */
  scanLocal(input: RiskScanInput): DetectedRisk[] {
    const risks: DetectedRisk[] = [];
    const today = new Date(input.today ?? Date.now());

    // [Bug #1] Cases loop
    for (const c of input.cases ?? []) {
      if (!c.summary && c.status !== 'closed') {
        risks.push({
          type: 'no_summary',
          severity: 'medium',
          description: `Дело «${c.name}» не имеет AI-сводки — контекст может быть потерян`,
          entityId: c.id,
          entityName: c.name,
          suggestedAction: 'Откройте дело и запустите генерацию AI-сводки',
        });
      }

      if (c.deadline && c.status !== 'closed') {
        const dl = new Date(c.deadline);
        const diffDays = Math.ceil((dl.getTime() - today.getTime()) / 86_400_000);
        risks.push(...buildDeadlineRisks(diffDays, c.name, c.deadline, c.id, 'делу'));
      }
    }

    // [Bug #1] Appeals loop — проверяем дедлайны и no_summary для апелляций
    for (const a of input.appeals ?? []) {
      if (!a.summary) {
        risks.push({
          type: 'no_summary',
          severity: 'medium',
          description: `Апелляция «${a.name}» не имеет AI-сводки — контекст может быть потерян`,
          entityId: a.id,
          entityName: a.name,
          suggestedAction: 'Откройте апелляцию и запустите генерацию AI-сводки',
        });
      }

      if (a.deadline) {
        const dl = new Date(a.deadline);
        const diffDays = Math.ceil((dl.getTime() - today.getTime()) / 86_400_000);
        risks.push(...buildDeadlineRisks(diffDays, a.name, a.deadline, a.id, 'апелляции'));
      }
    }

    return risks;
  },
};

/** [Bug #2] Построение рисков по дедлайнам с правильным порогом severity */
function buildDeadlineRisks(
  diffDays: number,
  name: string,
  deadline: string,
  id: number,
  entityType: 'делу' | 'апелляции',
): DetectedRisk[] {
  const result: DetectedRisk[] = [];

  if (diffDays < 0) {
    result.push({
      type: 'missed_deadline',
      severity: 'critical',
      description: `Срок по ${entityType} «${name}» истёк ${Math.abs(diffDays)} дн. назад (${deadline})`,
      entityId: id,
      entityName: name,
      suggestedAction: 'Уточните статус, уведомите клиента и закройте или перенесите срок',
    });
  } else if (diffDays === 0) {
    result.push({
      type: 'missed_deadline',
      severity: 'critical',
      description: `Срок по ${entityType} «${name}» истекает сегодня!`,
      entityId: id,
      entityName: name,
      suggestedAction: 'Немедленно проверьте готовность документов',
    });
  } else if (diffDays <= 1) {
    // [Bug #2] Порог critical: завтра = critical (diffDays <= 1), не <= 2
    result.push({
      type: 'missed_deadline',
      severity: 'critical',
      description: `Срок по ${entityType} «${name}» завтра (${deadline})`,
      entityId: id,
      entityName: name,
      suggestedAction: 'Подготовьте документы срочно',
    });
  } else if (diffDays <= 7) {
    result.push({
      type: 'missed_deadline',
      severity: 'high',
      description: `Срок по ${entityType} «${name}» через ${diffDays} дн. (${deadline})`,
      entityId: id,
      entityName: name,
      suggestedAction: 'Подготовьте документы заблаговременно',
    });
  } else if (diffDays <= 30) {
    result.push({
      type: 'missed_deadline',
      severity: 'medium',
      description: `Приближающийся срок по ${entityType} «${name}» — ${diffDays} дн. (${deadline})`,
      entityId: id,
      entityName: name,
      suggestedAction: 'Запланируйте подготовку документов',
    });
  }

  return result;
}

/** Дедупликация: убираем из llmRisks те, что уже есть в localRisks по entityId+type */
function mergeRisks(local: DetectedRisk[], llm: DetectedRisk[]): DetectedRisk[] {
  const localKeys = new Set(local.map((r) => `${r.type}:${r.entityId ?? ''}`));
  const uniqueLlm = llm.filter((r) => !localKeys.has(`${r.type}:${r.entityId ?? ''}`));
  return [...local, ...uniqueLlm].sort((a, b) => {
    const order = { critical: 0, high: 1, medium: 2, low: 3 };
    return order[a.severity] - order[b.severity];
  });
}
