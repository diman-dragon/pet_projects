// src/services/rag/IndexerService.ts
//
// ИТЕРАЦИЯ 2: DocumentIndexer — единая точка индексации.
// IndexerService сохраняет обратную совместимость через реэкспорт.

import { forceReembedAll } from './indexer/forceReembedAll';
import { indexFolder } from './indexer/indexFolder';
import { indexSingleFile } from './indexer/indexSingleFile';
import { DocumentIndexer } from '../search/DocumentIndexer';

export const IndexerService = {
  indexFile: indexSingleFile,
  indexFast: DocumentIndexer.indexFast,
  indexFull: DocumentIndexer.indexFull,
  forceReembedAll,
  indexFolder,
};

export { DocumentIndexer };
export type { IndexOptions, IndexResult } from './indexer/types';
