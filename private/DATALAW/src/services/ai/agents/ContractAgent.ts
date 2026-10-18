// src/services/ai/agents/ContractAgent.ts
// Анализ договора — возвращает структурированный JSON

import { aiApi } from '../../backend/aiApi';
import { getActiveChatModel } from '../config';
import { AppLanguageService } from '../../config/AppLanguageService';
import { getContextParams } from '../ModelContextProfile';
import { createLogger } from '../../logger/Logger';

const log = createLogger('ContractAgent');

export interface ContractAnalysisResult {
  contractType: string;
  parties: string[];
  subject: string;
  keyTerms: string[];
  obligations: { party: string; duties: string[] }[];
  deadlines: string[];
  penalties: string[];
  risks: string[];
  missingClauses: string[];
  recommendedActions: string[];
  confidence: number;
}

// Промпт вынесен сюда (не импортируется из contract.prompt.ts) чтобы избежать дублирования
// и чтобы агент был самодостаточным.
const CONTRACT_SYSTEM_PROMPT = `Ты — юридический аналитик договоров платформы Dorlane.
Задача: провести строгий анализ предоставленного договора и вернуть результат СТРОГО в формате JSON.
Никакого текста до или после JSON. Никаких markdown-блоков (без \`\`\`json).

Структура ответа (строгая, все поля обязательны):
{
  "contractType": "тип договора по ГК РФ (купля-продажа, возмездное оказание услуг, аренда, подряд, займ и т.д.)",
  "parties": ["полное наименование или ФИО стороны 1 из текста", "сторона 2 ..."],
  "subject": "предмет договора одним конкретным предложением",
  "keyTerms": [
    "конкретное ключевое условие: цена, срок, объём, качество, порядок расчётов — с цифрами если есть"
  ],
  "obligations": [
    {
      "party": "наименование стороны точно как в договоре",
      "duties": ["конкретное обязательство со ссылкой на пункт договора если возможно"]
    }
  ],
  "deadlines": [
    "конкретный срок с контекстом: 'Оплата — в течение 5 банковских дней с момента подписания акта (п. 4.2)'"
  ],
  "penalties": [
    "санкция с размером: 'Неустойка — 0,1% от суммы за каждый день просрочки (п. 6.1)'"
  ],
  "risks": [
    "конкретный правовой риск со ссылкой на пункт или отсутствующее условие"
  ],
  "missingClauses": [
    "существенное условие, обязательное по ГК РФ для данного типа договора, которое отсутствует в тексте"
  ],
  "recommendedActions": [
    "конкретное действие: 'Добавить пункт об ответственности за нарушение конфиденциальности'"
  ],
  "confidence": число от 0.0 до 1.0
}

ПРАВИЛА:
1. Анализируй ТОЛЬКО то, что есть в тексте. Не придумывай условий.
2. keyTerms — конкретные факты с цифрами, а не общие категории.
3. risks — с указанием конкретного пункта договора или отсутствующего условия.
4. missingClauses — только обязательные существенные условия по ГК РФ для данного типа.
5. confidence: снижай если текст неполный, обрезанный или нечитаемый.
6. Если документ не является договором — верни contractType: "не является договором", confidence: 0.1.`;

export const ContractAgent = {
  async analyze(
    contractText: string,
    onToken?: (t: string) => void,
  ): Promise<ContractAnalysisResult> {
    const model = await getActiveChatModel();
    if (!model) throw new Error('ContractAgent: модель не выбрана');

    const langInstruction = await AppLanguageService.getPromptInstruction();
    const ctxParams = getContextParams('contract');

    const messages = [
      {
        role: 'system' as const,
        content: `${langInstruction}\n\n${CONTRACT_SYSTEM_PROMPT}`,
      },
      {
        role: 'user' as const,
        content: `Договор для анализа:\n\n${contractText.slice(0, 14000)}`,
      },
    ];

    // [Bug #5] Исправлен race condition: используем response.content вместо event accumulation
    const response = await aiApi.chat(model, messages, false, ctxParams);
    const fullText = response.content;

    // Для плавности UI разбиваем ответ на токены и вызываем onToken
    if (onToken && fullText) {
      const chunkSize = 32; // ~8 слов за тик
      for (let i = 0; i < fullText.length; i += chunkSize) {
        const chunk = fullText.slice(i, i + chunkSize);
        onToken(chunk);
        // Небольшая задержка для визуального эффекта
        if (i < fullText.length - chunkSize) {
          await new Promise(r => setTimeout(r, 8));
        }
      }
    }

    try {
      const clean = fullText
        .replace(/```json\s*/gi, '')
        .replace(/```\s*/g, '')
        .trim();
      const start = clean.indexOf('{');
      const end = clean.lastIndexOf('}');
      const jsonStr = start !== -1 && end > start ? clean.slice(start, end + 1) : clean;
      return JSON.parse(jsonStr) as ContractAnalysisResult;
    } catch (err) {
      log.error('ContractAgent: failed to parse JSON', { err, textLength: fullText.length });
      return {
        contractType: 'Неопределён (ошибка парсинга)',
        parties: [],
        subject: fullText.slice(0, 500),
        keyTerms: [],
        obligations: [],
        deadlines: [],
        penalties: [],
        risks: ['Не удалось разобрать структуру ответа ИИ'],
        missingClauses: [],
        recommendedActions: ['Проверьте документ вручную или повторите анализ'],
        confidence: 0,
      };
    }
  },
};
