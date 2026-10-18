// src/services/ai/agents/CourtPrecedentAnalyzerAgent.ts
// Агент для анализа судебной практики и поиска прецедентов
//
// FIX: добавлен system prompt с языковой инструкцией + LAILA_ANTI_HALLUCINATION_RULES.
// Предыдущая версия отправляла ТОЛЬКО user-сообщение — LLM не имел идентичности,
// языковой директивы и правил антигаллюцинации.
// Агент с инструкцией «найди 5-10 дел» без RAG-контекста БУДЕТ галлюцинировать
// реальные номера дел (А40-12345/2023 и т.п.). Промпт переформулирован:
// без knowledgeBaseContext агент описывает стратегию, НЕ выдумывает решения.

import { aiApi } from '../../backend/aiApi';
import { getActiveChatModel } from '../config';
import { AppLanguageService } from '../../config/AppLanguageService';
import { getContextParams } from '../ModelContextProfile';
import { LAILA_IDENTITY_HEADER, LAILA_ANTI_HALLUCINATION_RULES } from '../prompts/laila.identity';
import { createLogger } from '../../logger/Logger';

const log = createLogger('CourtPrecedentAnalyzerAgent');

export interface PrecedentAnalysis {
    relevantCases: Array<{
        caseNumber: string;
        court: string;
        date: string;
        summary: string;
        relevanceScore: number;
        keyPoints: string[];
        citation: string;
    }>;
    legalPrinciples: string[];
    riskAssessment: {
        favorable: number;
        unfavorable: number;
        neutral: number;
    };
    recommendations: string[];
    confidence: number;
}

export interface PrecedentQuery {
    legalIssue: string;
    jurisdiction?: string;
    courtLevel?: 'district' | 'regional' | 'supreme' | 'constitutional';
    timeFrame?: { from: string; to: string };
    caseType?: string;
    // Если передан — агент анализирует только его. Без него — не называет конкретных дел.
    knowledgeBaseContext?: string;
}

const PRECEDENT_SYSTEM_PROMPT = `${LAILA_IDENTITY_HEADER}

${LAILA_ANTI_HALLUCINATION_RULES}

# РОЛЬ
Эксперт по судебной практике. Анализируешь правовые вопросы и помогаешь
выстроить стратегию поиска прецедентов.

# КРИТИЧЕСКИ ВАЖНО
Если контекст из базы знаний НЕ предоставлен — НЕ называй конкретные
номера дел, суды, даты. Без источника это галлюцинация.
Вместо этого: опиши правовые принципы и стратегию поиска.
Если контекст предоставлен — работай ТОЛЬКО с ним.

# ФОРМАТ — строгий JSON, без markdown:
{
  "relevantCases": [],
  "legalPrinciples": ["принцип 1"],
  "riskAssessment": {"favorable": 0.0, "unfavorable": 0.0, "neutral": 1.0},
  "recommendations": ["рекомендация"],
  "confidence": 0.0
}
Без контекста: relevantCases = [], confidence не выше 0.3.`;

export async function analyzePrecedents(query: PrecedentQuery): Promise<PrecedentAnalysis> {
    const model = await getActiveChatModel();
    const ctxParams = getContextParams('analyze');
    const langInstruction = await AppLanguageService.getPromptInstruction();

    const jurisdiction = query.jurisdiction || 'Российская Федерация';
    const courtLevel = query.courtLevel || 'все уровни';
    const timeFrame = query.timeFrame
        ? `с ${query.timeFrame.from} по ${query.timeFrame.to}`
        : 'за последние 5 лет';

    let userContent = `Правовой вопрос: "${query.legalIssue}"\n\nПараметры:\n- Юрисдикция: ${jurisdiction}\n- Уровень суда: ${courtLevel}\n- Период: ${timeFrame}`;
    if (query.caseType) userContent += `\n- Тип дел: ${query.caseType}`;

    if (query.knowledgeBaseContext) {
        userContent += `\n\n=== КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ ===\n${query.knowledgeBaseContext}\n===\n\nАнализируй ТОЛЬКО документы выше.`;
    } else {
        userContent += `\n\nБаза знаний не предоставлена. Опиши правовые принципы и стратегию поиска без конкретных номеров дел.`;
    }

    try {
        const response = await aiApi.chat(model, [
            { role: 'system', content: `${langInstruction}\n\n${PRECEDENT_SYSTEM_PROMPT}` },
            { role: 'user', content: userContent },
        ], false, {
            ...ctxParams,
            temperature: 0.1,
            num_predict: 2048,
        });

        const result = parsePrecedentResponse(response.content);
        log.info('Court precedent analysis completed', {
            query: query.legalIssue,
            casesFound: result.relevantCases.length,
            confidence: result.confidence,
        });
        return result;
    } catch (error) {
        log.error('Court precedent analysis failed', { query, error });
        throw new Error(`Failed to analyze court precedents: ${error}`);
    }
}

function parsePrecedentResponse(response: string): PrecedentAnalysis {
    try {
        const jsonMatch = response.match(/\{[\s\S]*\}/);
        if (!jsonMatch) throw new Error('No JSON in response');

        const parsed = JSON.parse(jsonMatch[0]);
        if (!Array.isArray(parsed.relevantCases)) parsed.relevantCases = [];

        if (parsed.riskAssessment) {
            const total = (parsed.riskAssessment.favorable || 0) +
                (parsed.riskAssessment.unfavorable || 0) +
                (parsed.riskAssessment.neutral || 0);
            if (total > 0) {
                parsed.riskAssessment.favorable = (parsed.riskAssessment.favorable || 0) / total;
                parsed.riskAssessment.unfavorable = (parsed.riskAssessment.unfavorable || 0) / total;
                parsed.riskAssessment.neutral = (parsed.riskAssessment.neutral || 0) / total;
            }
        }

        return parsed as PrecedentAnalysis;
    } catch (error) {
        log.warn('Failed to parse precedent response', { error });
        return {
            relevantCases: [],
            legalPrinciples: [],
            riskAssessment: { favorable: 0, unfavorable: 0, neutral: 1 },
            recommendations: ['Не удалось проанализировать судебную практику'],
            confidence: 0,
        };
    }
}
