import { aiApi } from '../../backend/aiApi';
import { AppLanguageService } from '../../config/AppLanguageService';
import { createLogger } from '../../logger/Logger';
import { getContextParams } from '../ModelContextProfile';
import { getActiveChatModel } from '../config';

const log = createLogger('DeadlineTrackerAgent');

export interface TrackedDeadline {
  title: string;
  dueDate: string | null;
  sourceText: string;
  actionRequired: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  confidence: number;
}

export interface DeadlineTrackerResult {
  deadlines: TrackedDeadline[];
  summary: string;
  confidence: number;
}

const DEADLINE_TRACKER_PROMPT = `Ты — юридический аналитик сроков платформы Dorlane.
Извлеки из текста все сроки, дедлайны и процессуальные даты.

Верни строго JSON:
{
  "summary": "краткое резюме по срокам",
  "deadlines": [
    {
      "title": "краткое название срока",
      "dueDate": "YYYY-MM-DD или null если дата не определена точно",
      "sourceText": "дословный фрагмент или краткая цитата из текста",
      "actionRequired": "что нужно сделать к сроку",
      "priority": "critical | high | medium | low",
      "confidence": число от 0.0 до 1.0
    }
  ],
  "confidence": число от 0.0 до 1.0
}

Правила:
1. Не придумывай даты, если они не указаны явно.
2. Если срок относительный ("в течение 10 дней"), а опорной даты нет — dueDate = null.
3. В summary кратко опиши самые срочные обязательства.
4. Если сроков нет — deadlines = [] и summary = "Сроки не обнаружены".`;

export const DeadlineTrackerAgent = {
  async track(text: string, contextLabel?: string): Promise<DeadlineTrackerResult> {
    const model = await getActiveChatModel();
    if (!model) throw new Error('DeadlineTrackerAgent: модель не выбрана');

    const langInstruction = await AppLanguageService.getPromptInstruction();
    const messages = [
      { role: 'system' as const, content: `${langInstruction}\n\n${DEADLINE_TRACKER_PROMPT}` },
      {
        role: 'user' as const,
        content: `Контекст: ${contextLabel || 'document'}\n\nТекст для анализа:\n${text.slice(0, 16000)}`,
      },
    ];

    try {
      const response = await aiApi.chat(model, messages, false, getContextParams('analyze'));
      const clean = response.content.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim();
      const start = clean.indexOf('{');
      const end = clean.lastIndexOf('}');
      const parsed = JSON.parse(start !== -1 && end > start ? clean.slice(start, end + 1) : clean);

      return {
        deadlines: Array.isArray(parsed.deadlines) ? parsed.deadlines : [],
        summary: typeof parsed.summary === 'string' ? parsed.summary : 'Сроки не обнаружены',
        confidence: typeof parsed.confidence === 'number' ? parsed.confidence : 0.6,
      };
    } catch (error) {
      log.error('DeadlineTrackerAgent: parse failed', error);
      return {
        deadlines: [],
        summary: 'Сроки не обнаружены',
        confidence: 0,
      };
    }
  },
};
