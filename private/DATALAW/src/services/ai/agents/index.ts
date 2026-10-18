// src/services/ai/agents/index.ts
// [3.2] Экспорт всех агентов

export { ContractAgent } from './ContractAgent';
export type { ContractAnalysisResult } from './ContractAgent';

export { RiskScannerAgent } from './RiskScannerAgent';
export type { RiskScanResult, DetectedRisk, RiskSeverity, RiskScanInput } from './RiskScannerAgent';

export { DocumentClassifierAgent } from './DocumentClassifierAgent';
export type { ClassificationResult, DocumentClass } from './DocumentClassifierAgent';

export { DefenseStrategyAgent } from './DefenseStrategyAgent';
export type { StrategyStep } from './DefenseStrategyAgent';

export { DocumentDraftAgent } from './DocumentDraftAgent';

export { DeadlineTrackerAgent } from './DeadlineTrackerAgent';
export type { DeadlineTrackerResult, TrackedDeadline } from './DeadlineTrackerAgent';

export { KnowledgeSearchAgent } from './KnowledgeSearchAgent';
export type { SearchResult } from '../../rag/UnifiedSearchService'; // Типы для KnowledgeSearchAgent

export { LegalResearchAgent } from './LegalResearchAgent';
export type { LegalResearchResult } from './LegalResearchAgent';

export { analyzePrecedents, type PrecedentAnalysis, type PrecedentQuery } from './CourtPrecedentAnalyzerAgent';

// Knowledge Graph
export { KnowledgeGraphAgent } from './KnowledgeGraphAgent';
export type {
  KgEntity,
  KgRelation,
  KgEntityType,
  KgRelationType,
  KgExtractionResult,
} from './KnowledgeGraphAgent';
