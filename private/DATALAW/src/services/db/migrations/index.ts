// src/services/db/migrations/index.ts
export type { Migration } from './types';

import { BASE_MIGRATIONS } from './base';
import { CLIENT_MIGRATIONS } from './clients';
import { HUMAN_RIGHTS_MIGRATIONS } from './human_rights';
import { KNOWLEDGE_MIGRATIONS } from './knowledge';
import { KNOWLEDGE_GRAPH_MIGRATIONS } from './knowledge_graph';
import { LEGAL_CORPUS_MIGRATIONS } from './legalCorpus';
import { PII_VAULT_MIGRATIONS } from './piiVault';

export const MIGRATIONS = [
  ...BASE_MIGRATIONS,
  ...KNOWLEDGE_MIGRATIONS,
  ...CLIENT_MIGRATIONS,
  ...HUMAN_RIGHTS_MIGRATIONS,
  ...KNOWLEDGE_GRAPH_MIGRATIONS,
  ...PII_VAULT_MIGRATIONS,
  ...LEGAL_CORPUS_MIGRATIONS,
];
