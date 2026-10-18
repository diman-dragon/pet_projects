// src/services/ai/prompts/index.ts

export { WIDGET_SYSTEM_PROMPT } from './widget.prompt';
export { ANALYZE_SYSTEM_PROMPT } from './analyze.prompt';
export {
  KNOWLEDGE_SYSTEM_PROMPT,
  KNOWLEDGE_RAG_INSTRUCTION,
  KNOWLEDGE_EMPTY_CONTEXT_MESSAGE,      // [3.1] сообщение о пустой базе
} from './knowledge.prompt';
export { DOSSIER_SYSTEM_PROMPT } from './dossier.prompt';
export { SYSTEMATIZE_SYSTEM_PROMPT } from './systematize.prompt';
// [3.2] Новые промпты агентов
export { CONTRACT_SYSTEM_PROMPT } from './contract.prompt';
export { RISK_SCANNER_SYSTEM_PROMPT } from './riskScanner.prompt';
export { STRATEGY_SYSTEM_PROMPT } from './strategy.prompt';
export { DRAFT_SYSTEM_PROMPT } from './draft.prompt';
// Knowledge Graph
export { KNOWLEDGE_GRAPH_SYSTEM_PROMPT } from './knowledgeGraph.prompt';
