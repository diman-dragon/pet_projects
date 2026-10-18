// src/services/rag/llmReranker.ts
//
// Cross-encoder reranker via Rust Tauri command (pure-Rust BM25 + position penalty).
// Performance: <5ms for 20 candidates (no LLM, no network, no model file).
//
// Hard-cutoff logic:
//   If the best final_score < HARD_CUTOFF_THRESHOLD the query likely has no relevant
//   content in the knowledge base → return [] to trigger "no information" message.
//   Threshold 0.10 is calibrated for the pure-Rust scorer.
//
// Relative cutoff:
//   RELATIVE_CUTOFF_RATIO = 0.20 — keeps more candidates from small knowledge bases.
//
// FIX v2: батчи обрабатываются параллельно через Promise.all вместо последовательного цикла.
// При 30 кандидатах (2 батча по 15) даёт x1.5-2 ускорение на invoke-round-trip.

import { invoke } from '@tauri-apps/api/core';
import { createLogger } from '../logger/Logger';
import { getRagConfig } from './config';
import type { RetrievalCandidate } from './hybridRetrievalService';

const log = createLogger('CrossEncoderReranker');

export interface LLMRerankOptions {
    query: string;
    candidates: RetrievalCandidate[];
    maxCandidates?: number;
}

export interface LLMRerankResult {
    chunkId: number;
    score: number;
    positionPenalty: number;
    finalScore: number;
}

const HARD_CUTOFF_THRESHOLD = 0.10;
const RELATIVE_CUTOFF_RATIO = 0.20;

async function processBatch(
    query: string,
    batch: RetrievalCandidate[],
    maxCandidates: number
): Promise<LLMRerankResult[]> {
    const validatedCandidates = batch.map(c => ({
        chunkId: c.chunkId ?? 0,
        libraryId: c.libraryId ?? 0,
        title: c.title ?? '',
        category: c.category ?? '',
        filePath: c.filePath ?? '',
        chunkText: c.chunkText ?? '',
        chunkIndex: c.chunkIndex ?? 0,
        score: c.rrfScore ?? 0,
        matchType: c.matchType ?? 'hybrid',
    }));

    const rerankResults: LLMRerankResult[] = await invoke('rag_rerank_candidates', {
        request: {
            query,
            candidates: validatedCandidates,
            maxCandidates,
        }
    });

    return rerankResults || [];
}

export async function llmRerank(
    options: LLMRerankOptions
): Promise<RetrievalCandidate[]> {
    const config = getRagConfig();
    const { query, candidates, maxCandidates = Math.max(config.limits.maxCandidates, 25) } = options;

    if (candidates.length === 0) return [];

    if (candidates.length <= 3) {
        log.debug('Skipping rerank for small candidate set', { count: candidates.length });
        return candidates
            .sort((a, b) => (b.fastRerankScore || b.rrfScore) - (a.fastRerankScore || a.rrfScore))
            .slice(0, maxCandidates)
            .map(c => ({ ...c, llmRerankScore: c.fastRerankScore ?? c.rrfScore }));
    }

    const batchSize = 15;
    const candidatesToRank = candidates.slice(0, Math.min(candidates.length, 30));

    log.debug('Starting Rust rerank with parallel batching', {
        query,
        totalCandidates: candidatesToRank.length,
        batchSize,
        maxCandidates
    });

    try {
        // FIX: параллельная обработка батчей вместо последовательного цикла.
        // invoke('rag_rerank_candidates') — чистый Rust, не блокирует JS thread,
        // параллельный запуск безопасен и даёт x1.5-2 ускорение.
        const batches: RetrievalCandidate[][] = [];
        for (let i = 0; i < candidatesToRank.length; i += batchSize) {
            batches.push(candidatesToRank.slice(i, i + batchSize));
        }

        const batchResults = await Promise.all(
            batches.map(batch => processBatch(query, batch, maxCandidates))
        );

        const allRerankResults: LLMRerankResult[] = batchResults.flat();

        if (allRerankResults.length === 0) {
            log.warn('Rust reranker returned empty results, using fast rerank fallback');
            return fastRerankFallback(candidates, maxCandidates);
        }

        const bestScore = Math.max(...allRerankResults.map(r => r.finalScore));

        if (bestScore < HARD_CUTOFF_THRESHOLD) {
            log.warn('Hard cutoff triggered — no relevant content', {
                bestScore,
                threshold: HARD_CUTOFF_THRESHOLD,
                query,
            });
            return [];
        }

        const relativeCutoff = bestScore * RELATIVE_CUTOFF_RATIO;
        const filteredByRelative = allRerankResults.filter(r => r.finalScore >= relativeCutoff);

        const candidateMap = new Map(candidatesToRank.map(c => [c.chunkId, c]));
        const ranked = filteredByRelative
            .map(r => {
                const c = candidateMap.get(r.chunkId);
                if (!c) return null;
                return {
                    ...c,
                    llmRerankScore: r.finalScore,
                } as RetrievalCandidate;
            })
            .filter((c): c is RetrievalCandidate => c !== null);

        log.debug('Rust rerank complete', {
            ranked: ranked.length,
            bestScore,
            worstScore: ranked[ranked.length - 1]?.llmRerankScore,
        });

        return ranked.slice(0, maxCandidates);

    } catch (error) {
        console.error('[🦀 RUST RERANK] FAILED:', error);
        log.error('Rust reranker failed, falling back to fast rerank', error);
        return fastRerankFallback(candidates, maxCandidates);
    }
}

function fastRerankFallback(
    candidates: RetrievalCandidate[],
    maxCandidates: number,
): RetrievalCandidate[] {
    return candidates
        .sort((a, b) => (b.fastRerankScore || 0) - (a.fastRerankScore || 0))
        .slice(0, maxCandidates)
        .map(c => ({ ...c, llmRerankScore: c.fastRerankScore ?? c.rrfScore }));
}
