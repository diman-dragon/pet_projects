import { LEGAL_CORPUS_SCHEMA } from '../../../constants/schemas/LegalCorpusSchema';
import type { Migration } from './types';

export const LEGAL_CORPUS_MIGRATIONS: Migration[] = [
  {
    version: 1,
    key: 'legal_corpus',
    description: 'Create legal corpus schema',
    sql: LEGAL_CORPUS_SCHEMA,
  },
];
