// src/services/logger/SearchLogger.ts
//
// ИТЕРАЦИЯ 3: Расширенная телеметрия поиска.
//
// Новые поля в search events:
//   - query hash (sha-like short hash для анонимного отслеживания)
//   - queryLang  — определённый язык запроса
//   - queryType  — тип (exact_identifier | name | date_range | free_text)
//   - latency стадий (vectorize_ms, fts_ms, merge_ms, agent_ms)
//   - counts (vector_hits, fts_hits, merged_count, final_chunks)
//   - selectedResult — id результата, который выбрал пользователь
//
// Обратная совместимость: все старые методы сохранены без изменений.

import { writeTextFile, readFile, exists, mkdir } from '@tauri-apps/plugin-fs';
import { join } from '@tauri-apps/api/path';
import { AppPaths } from '../config/AppPaths';
import { shortHash } from '../utils/hash';

export interface SearchLogEntry {
  timestamp: string;
  sessionId: string;
  stage: string;
  data: unknown;
}

// ── Новые типы для расширенной телеметрии (итерация 3) ───────────────────────

export interface SearchTelemetryEvent {
  queryHash: string;
  queryLang?: string;
  queryType?: 'exact_identifier' | 'name' | 'date_range' | 'free_text';
  latency?: {
    total_ms?: number;
    vectorize_ms?: number;
    fts_ms?: number;
    merge_ms?: number;
    agent_ms?: number;
  };
  counts?: {
    vector_hits?: number;
    fts_hits?: number;
    merged_count?: number;
    final_chunks?: number;
  };
  selectedResultId?: string;
}

// ── Per-session logger ────────────────────────────────────────────────────────

class SessionLogger {
  private logFilePath: string | null = null;
  private sessionId: string;
  private chatId: string;
  private initPromise: Promise<void>;
  private buffer: string[] = [];
  private pendingWrites: Promise<void> = Promise.resolve();

  constructor(chatId: string = 'default') {
    this.chatId = chatId;
    this.sessionId = this.generateSessionId();
    this.initPromise = this.init();
  }

  private generateSessionId(): string {
    const now = new Date();
    const date = now.toISOString().split('T')[0];
    const time = now.toTimeString().split(' ')[0].replace(/:/g, '-');
    return `${date}_${time}_${Math.random().toString(36).slice(2, 6)}`;
  }

  private async init(): Promise<void> {
    try {
      const kbPath = AppPaths.getKbPath();
      if (!kbPath) {
        throw new Error('KB path not initialized');
      }

      const logsDir = await join(kbPath, 'logs');
      try {
        const dirExists = await exists(logsDir);
        if (!dirExists) {
          await mkdir(logsDir);
        }
      } catch {
        // Directory might already exist
      }

      const safeChatId = this.chatId.replace(/[^a-zA-Z0-9_-]/g, '_').substring(0, 30);
      const filename = `search_${safeChatId}_${this.sessionId}.log`;
      this.logFilePath = await join(logsDir, filename);

      const header = `════════════════════════════════════════════════════════════════════════════════
🔍 SEARCH PIPELINE LOG [ITERATION 3]
Chat ID: ${this.chatId}
Session ID: ${this.sessionId}
Started: ${new Date().toISOString()}
Log File: ${this.logFilePath}
════════════════════════════════════════════════════════════════════════════════

`;
      await writeTextFile(this.logFilePath, header);
    } catch (error) {
      console.error('[❌ SEARCH LOGGER] Failed to initialize:', error);
      throw error;
    }
  }

  public async ensureInitialized(): Promise<void> {
    await this.initPromise;
  }

  private formatLogEntry(entry: SearchLogEntry): string {
    const timestamp = entry.timestamp;
    const stage = entry.stage.padEnd(25);

    let dataStr: string;
    try {
      dataStr = JSON.stringify(entry.data, null, 2);
    } catch {
      dataStr = String(entry.data);
    }

    return `
────────────────────────────────────────────────────────────────────────────────
[${timestamp}] [${stage}]
────────────────────────────────────────────────────────────────────────────────
${dataStr}
────────────────────────────────────────────────────────────────────────────────
`;
  }

  private async appendToLogFile(content: string): Promise<void> {
    if (!this.logFilePath) return;

    this.pendingWrites = this.pendingWrites.then(async () => {
      try {
        let existingContent = '';
        try {
          const existingBytes = await readFile(this.logFilePath!);
          existingContent = new TextDecoder().decode(existingBytes);
        } catch {
          // File doesn't exist or is empty
        }
        await writeTextFile(this.logFilePath!, existingContent + content);
      } catch (error) {
        console.error('[❌ SEARCH LOGGER] Failed to append:', error);
      }
    });

    await this.pendingWrites;
  }

  public async log(stage: string, data: unknown): Promise<void> {
    await this.ensureInitialized();

    const entry: SearchLogEntry = {
      timestamp: new Date().toISOString(),
      sessionId: this.sessionId,
      stage,
      data,
    };

    const formatted = this.formatLogEntry(entry);

    if (this.logFilePath) {
      await this.appendToLogFile(formatted);
    } else {
      this.buffer.push(formatted);
    }
  }

  public async flushBuffer(): Promise<void> {
    await this.ensureInitialized();
    if (this.buffer.length === 0 || !this.logFilePath) return;
    const content = this.buffer.join('');
    this.buffer = [];
    await this.appendToLogFile(content);
  }

  public getLogFilePath(): string | null {
    return this.logFilePath;
  }

  public getSessionId(): string {
    return this.sessionId;
  }

  public async dispose(): Promise<void> {
    await this.flushBuffer();
  }
}

// ── Manager ───────────────────────────────────────────────────────────────────

class SearchLoggerManager {
  private loggers: Map<string, SessionLogger> = new Map();
  private currentChatId: string | null = null;

  public startChat(chatId: string): SessionLogger {
    const existing = this.loggers.get(chatId);
    if (existing) {
      existing.dispose();
    }
    const logger = new SessionLogger(chatId);
    this.loggers.set(chatId, logger);
    this.currentChatId = chatId;
    return logger;
  }

  public getLogger(chatId?: string): SessionLogger | null {
    const id = chatId || this.currentChatId;
    if (!id) return null;
    return this.loggers.get(id) || null;
  }

  public async logToChat(chatId: string, stage: string, data: unknown): Promise<void> {
    let logger = this.loggers.get(chatId);
    if (!logger) {
      logger = this.startChat(chatId);
    }
    await logger.log(stage, data);
  }

  public async log(stage: string, data: unknown, chatId?: string): Promise<void> {
    const id = chatId || this.currentChatId;
    if (!id) {
      console.warn('[⚠️ SEARCH LOGGER] No active chat, logging to console only:', stage, data);
      console.log(`[🔍 ${stage}]`, data);
      return;
    }
    await this.logToChat(id, stage, data);
  }

  public setCurrentChat(chatId: string): void {
    this.currentChatId = chatId;
    if (!this.loggers.has(chatId)) {
      this.startChat(chatId);
    }
  }

  public async disposeAll(): Promise<void> {
    for (const logger of this.loggers.values()) {
      await logger.dispose();
    }
    this.loggers.clear();
  }

  public getLogPaths(): Array<{ chatId: string; path: string | null }> {
    return Array.from(this.loggers.entries()).map(([chatId, logger]) => ({
      chatId,
      path: logger.getLogFilePath(),
    }));
  }
}

const manager = new SearchLoggerManager();

// ── Публичный API (с обратной совместимостью) ─────────────────────────────────

export const searchLogger = {
  async log(stage: string, data: unknown): Promise<void> {
    await manager.log(stage, data);
  },

  async logQuery(
    query: string,
    context?: { memory: number; queryLang?: string; queryType?: string },
  ): Promise<void> {
    await manager.log('QUERY_INPUT', {
      queryHash: shortHash(query),
      queryLength: query.length,
      wordCount: query.split(/\s+/).length,
      hasQuestionMark: query.includes('?'),
      memoryContextMessages: context?.memory || 0,
      queryLang: context?.queryLang ?? 'unknown',
      queryType: context?.queryType ?? 'unknown',
      timestamp: Date.now(),
    });
  },

  /**
   * НОВЫЙ (итерация 3): логирует расширенную телеметрию поискового события.
   */
  async logSearchTelemetry(
    query: string,
    telemetry: SearchTelemetryEvent,
  ): Promise<void> {
    await manager.log('SEARCH_TELEMETRY', {
      queryHash: shortHash(query),
      queryLang: telemetry.queryLang ?? 'unknown',
      queryType: telemetry.queryType ?? 'unknown',
      latency: telemetry.latency ?? {},
      counts: telemetry.counts ?? {},
      selectedResultId: telemetry.selectedResultId ?? null,
      timestamp: Date.now(),
    });
  },

  /**
   * НОВЫЙ (итерация 3): логирует выбор результата пользователем.
   */
  async logResultSelected(
    query: string,
    resultId: string,
    resultSection: string,
    queryLang?: string,
  ): Promise<void> {
    await manager.log('RESULT_SELECTED', {
      queryHash: shortHash(query),
      resultId,
      resultSection,
      queryLang: queryLang ?? 'unknown',
      timestamp: Date.now(),
    });
  },

  async logQueryUnderstanding(
    query: string,
    understanding: {
      intent: string;
      entities: string[];
      needsExpansion: boolean;
      complexity: string;
    },
  ): Promise<void> {
    await manager.log('STAGE_1_QUERY_UNDERSTANDING', {
      queryHash: shortHash(query),
      intent: understanding.intent,
      entities: understanding.entities,
      entityCount: understanding.entities.length,
      needsExpansion: understanding.needsExpansion,
      complexity: understanding.complexity,
    });
  },

  async logQueryRewrite(
    original: string,
    rewritten: {
      original: string;
      rewritten: string;
      expansions: string[];
      finalQuery: string;
    },
  ): Promise<void> {
    await manager.log('STAGE_2_QUERY_REWRITE', {
      queryHash: shortHash(original),
      rewrittenQuery: rewritten.rewritten,
      finalQuery: rewritten.finalQuery,
      expansions: rewritten.expansions,
      expansionCount: rewritten.expansions.length,
    });
  },

  async logHybridRetrieval(
    query: string,
    results: {
      vectorCount: number;
      ftsCount: number;
      finalCount: number;
      candidates: Array<{
        chunkId: number;
        libraryId: number;
        title: string;
        category: string;
        rrfScore: number;
        matchType: string;
        vectorScore?: number;
        ftsScore?: number;
        metadataScore?: number;
      }>;
    },
  ): Promise<void> {
    await manager.log('STAGE_3_HYBRID_RETRIEVAL', {
      queryHash: shortHash(query),
      searchResults: {
        vectorResults: results.vectorCount,
        ftsResults: results.ftsCount,
        fusedResults: results.finalCount,
        fusionMethod: 'RRF k=60',
      },
      topCandidates: results.candidates.slice(0, 15).map((c, idx) => ({
        rank: idx + 1,
        chunkId: c.chunkId,
        title: c.title,
        rrfScore: c.rrfScore.toFixed(4),
        matchType: c.matchType,
      })),
    });
  },

  async logFastRerank(
    input: { count: number },
    output: { count: number; topScore?: number },
    details: Array<{
      chunkId: number;
      title: string;
      termOverlapScore: number;
      embeddingSimilarityScore: number;
      metadataScore: number;
      lengthScore: number;
      combinedScore: number;
    }>,
  ): Promise<void> {
    await manager.log('STAGE_4_FAST_RERANK', {
      inputCandidates: input.count,
      outputCandidates: output.count,
      topScore: output.topScore?.toFixed(4) || 'N/A',
      rankingDetails: details.slice(0, 20).map((d) => ({
        chunkId: d.chunkId,
        title: d.title.substring(0, 50),
        scores: {
          termOverlap: d.termOverlapScore.toFixed(3),
          embeddingSim: d.embeddingSimilarityScore.toFixed(3),
          metadata: d.metadataScore.toFixed(3),
          combined: d.combinedScore.toFixed(3),
        },
      })),
    });
  },

  async logLLMRerank(
    input: { count: number },
    output: { count: number; topScore?: number },
    _llmResponse?: { ranking: Array<{ id: number; score: number }> },
    timing?: { duration: number },
  ): Promise<void> {
    await manager.log('STAGE_5_LLM_RERANK', {
      inputCandidates: input.count,
      outputCandidates: output.count,
      topScore: output.topScore?.toFixed(4) || 'N/A',
      timing: timing ? `${timing.duration}ms` : 'N/A',
    });
  },

  async logContextCompression(
    input: { count: number },
    output: {
      chunks: Array<{ chunkId: number; title: string; chunkText: string }>;
      totalTokens: number;
      compressionRatio: number;
      summary: string;
    },
  ): Promise<void> {
    await manager.log('STAGE_6_CONTEXT_COMPRESSION', {
      inputChunks: input.count,
      outputChunks: output.chunks.length,
      compressionRatio: output.compressionRatio.toFixed(2),
      totalTokens: output.totalTokens,
      summary: output.summary,
    });
  },

  async logAnswerGeneration(
    input: {
      query: string;
      context: Array<{ chunkId: number; title: string; chunkText: string }>;
    },
    output: {
      answer: string;
      sources: Array<{ index: number; title: string; excerpt: string }>;
      confidence: number;
    },
    timing?: { duration: number },
  ): Promise<void> {
    await manager.log('STAGE_7_ANSWER_GENERATION', {
      queryHash: shortHash(input.query),
      answerLength: output.answer.length,
      hasCitations: /\[\d+\]/.test(output.answer),
      sources: output.sources.map((s) => ({ index: s.index, title: s.title })),
      confidence: output.confidence.toFixed(4),
      timing: timing ? `${timing.duration}ms` : 'N/A',
    });
  },

  async logVerification(result: {
    isValid: boolean;
    confidence: number;
    issues: string[];
    groundedScore: number;
  }): Promise<void> {
    await manager.log('STAGE_8_VERIFICATION', {
      isValid: result.isValid,
      confidence: result.confidence.toFixed(4),
      groundedScore: result.groundedScore.toFixed(4),
      issues: result.issues,
      verdict: result.isValid ? '✅ PASSED' : '❌ FAILED',
    });
  },

  async logPipelineComplete(
    query: string,
    finalAnswer: string,
    metrics: {
      totalTime: number;
      stagesCompleted: number;
      totalChunksRetrieved: number;
      finalChunksUsed: number;
    },
  ): Promise<void> {
    await manager.log('PIPELINE_COMPLETE', {
      queryHash: shortHash(query),
      answerLength: finalAnswer.length,
      performance: {
        totalTimeMs: metrics.totalTime,
        totalTimeSeconds: (metrics.totalTime / 1000).toFixed(2),
        stagesCompleted: metrics.stagesCompleted,
      },
      retrievalStats: {
        totalChunksRetrieved: metrics.totalChunksRetrieved,
        finalChunksUsed: metrics.finalChunksUsed,
      },
    });

    const logger = manager.getLogger();
    if (logger) {
      await logger.dispose();
    }
  },

  async logError(stage: string, error: unknown, context?: unknown): Promise<void> {
    await manager.log('ERROR', {
      stage,
      error:
        error instanceof Error
          ? { message: error.message, stack: error.stack, name: error.name }
          : String(error),
      context,
      timestamp: Date.now(),
    });
  },

  getLogFilePath(): string | null {
    const logger = manager.getLogger();
    return logger?.getLogFilePath() || null;
  },

  setChat(chatId: string): void {
    manager.setCurrentChat(chatId);
  },

  startNewChat(chatId: string): void {
    manager.startChat(chatId);
  },

  dispose(): Promise<void> {
    return manager.disposeAll();
  },
};

export { manager as searchLoggerManager };

export const logSearch = (stage: string, data: unknown): void => {
  searchLogger.log(stage, data).catch(console.error);
};

export default searchLogger;
