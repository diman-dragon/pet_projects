// src/services/rag/fastReranker.ts
//
// Stage 4 Fast Reranker
//
// ИЗМЕНЕНИЯ v3:
// 1. TERM GATE: если термины запроса не встречаются в тексте чанка (overlap < 0.02),
//    финальный score умножается на 0.2.
// 2. HYBRID BONUS: чанки с matchType='hybrid' (vector+fts) получают +15% к итоговому score.
// 3. TITLE BOOST: если все токены запроса встречаются в title — +0.2 к metadata score.
// 4. FIX POSITION DECAY: penalty применяется только для chunkIndex > 5 (не для всех не-первых).
//    Статья в середине документа не должна получать штраф только за позицию.
// 5. MIN_FAST_RERANK_SCORE снижен до 0.01 — hard cutoff делается на уровне llmReranker.

import { createLogger } from '../logger/Logger';
import type { RetrievalCandidate } from './hybridRetrievalService';
import { stemTokenWithFallback } from './utils/stemming';

const log = createLogger('FastReranker');

export interface FastRerankOptions {
    query: string;
    candidates: RetrievalCandidate[];
    maxCandidates?: number;
    keywords?: string[];
}

const MAX_CANDIDATES = 20;
const MIN_FAST_RERANK_SCORE = 0.01;
// FIX: position decay только для глубоких чанков (>5), не для всех не-первых.
// Релевантная статья закона может быть в чанке 3-5 и не должна штрафоваться.
const POSITION_DECAY_THRESHOLD = 5;

function tokenizeWithStems(text: string): Set<string> {
    const result = new Set<string>();
    const raw = text.toLowerCase()
        .split(/\s+/)
        .filter(t => t.length > 2)
        .map(t => t.replace(/[^\wа-яё]/gi, ''));
    for (const t of raw) {
        result.add(t);
        const stem = stemTokenWithFallback(t);
        if (stem !== t) result.add(stem);
    }
    return result;
}

export async function fastRerank(
    options: FastRerankOptions,
): Promise<RetrievalCandidate[]> {
    const { query, candidates, maxCandidates = MAX_CANDIDATES, keywords = [] } = options;

    if (candidates.length === 0) return [];

    log.debug('Starting fast rerank', { query, candidateCount: candidates.length });

    const effectiveQuery = keywords.length > 0 ? `${query} ${keywords.join(' ')}` : query;
    const queryTerms = tokenizeWithStems(effectiveQuery);

    const scored = candidates.map((candidate) => {
        const chunkTerms = tokenizeWithStems(candidate.chunkText);

        // Term overlap (Jaccard on query terms)
        const intersection = new Set([...queryTerms].filter(t => chunkTerms.has(t)));
        const termOverlap = queryTerms.size > 0 ? intersection.size / queryTerms.size : 0;

        // Embedding similarity (from vector search)
        const embeddingScore = Math.max(0, candidate.vectorScore ?? termOverlap * 0.5);

        // Metadata score
        const metadataScore = calculateMetadataScore(queryTerms, candidate);

        // Base formula: 0.5 * term_overlap + 0.4 * embedding + 0.1 * metadata
        const baseScore = 0.5 * termOverlap + 0.4 * embeddingScore + 0.1 * metadataScore;

        // TERM GATE: если термины не встречаются в чанке — резко снижаем score.
        const termGate = termOverlap < 0.02 ? 0.2 : 1.0;

        // HYBRID BONUS: vector+fts совпадение надёжнее одного канала
        const hybridBonus = candidate.matchType === 'hybrid' ? 1.15 : 1.0;

        // FIX POSITION DECAY: штрафуем только глубокие чанки (>POSITION_DECAY_THRESHOLD).
        // chunkIndex 1-5 — нормальное начало/середина документа, не штрафуем.
        const positionDecay = candidate.chunkIndex > POSITION_DECAY_THRESHOLD ? 0.95 : 1.0;

        const finalScore = baseScore * termGate * hybridBonus * positionDecay;

        return { ...candidate, fastRerankScore: finalScore };
    });

    scored.sort((a, b) => (b.fastRerankScore || 0) - (a.fastRerankScore || 0));

    const filtered = scored.filter(c => (c.fastRerankScore || 0) >= MIN_FAST_RERANK_SCORE);
    const result = (filtered.length > 0 ? filtered : scored).slice(0, maxCandidates);

    log.debug('Fast rerank done', {
        input: candidates.length,
        output: result.length,
        topScore: result[0]?.fastRerankScore?.toFixed(4),
        bottomScore: result[result.length - 1]?.fastRerankScore?.toFixed(4),
    });

    return result;
}

function calculateMetadataScore(queryTerms: Set<string>, candidate: RetrievalCandidate): number {
    let score = 0;

    const titleTerms = tokenizeWithStems(candidate.title ?? '');
    const catTerms = tokenizeWithStems(candidate.category ?? '');
    const pathTerms = tokenizeWithStems(candidate.filePath ?? '');

    // Title hit — проверяем каждый токен запроса
    const titleHits = [...queryTerms].filter(t => t.length >= 3 && titleTerms.has(t)).length;
    if (titleHits > 0) {
        // TITLE BOOST: если все значимые токены запроса в title — сильный boost
        const significantQueryTerms = [...queryTerms].filter(t => t.length >= 4);
        const allInTitle = significantQueryTerms.length > 0 &&
            significantQueryTerms.every(t => titleTerms.has(t));
        score += allInTitle ? 0.6 : 0.3;
    }

    // Category hit
    const catHit = [...queryTerms].some(t => t.length >= 4 && catTerms.has(t));
    if (catHit) score += 0.2;

    // File path hit
    const pathHit = [...queryTerms].some(t => t.length >= 4 && pathTerms.has(t));
    if (pathHit) score += 0.15;

    return Math.min(1.0, score);
}
