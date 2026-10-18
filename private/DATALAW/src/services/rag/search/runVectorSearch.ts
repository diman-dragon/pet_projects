import type { SearchResult } from '../UnifiedSearchService';
import type { DbCapability } from './capability';
import { searchJsCosine } from './cosine';
import { searchVec0 } from './vec0';
import { searchVecScalar } from './vecScalar';
import { createLogger } from '../../logger/Logger';

const log = createLogger('VectorSearch');
const vecLog = (id: string, msg: string) => log.debug(`[VEC ${id}] ${msg}`);

// ИЗМЕНЕНИЕ: categoryFilter (raw string) заменён на category (string | undefined).
// Все search-функции принимают category напрямую и параметризуют его сами.
export async function runVectorSearch(
  db: any,
  capability: DbCapability,
  queryVector: Float32Array,
  limit: number,
  category: string | undefined,
  model: string | string[],
): Promise<SearchResult[]> {
  const vid = Math.random().toString(36).slice(2, 6);
  vecLog(vid, `capability=${capability}, limit=${limit}, category=${category ?? 'all'}, dim=${queryVector.length}`);

  const modelList = Array.isArray(model) ? model : [model];

  if (capability === 'vec0') {
    vecLog(vid, 'trying vec0...');
    const vec0Results = await searchVec0(db, queryVector, limit, category, modelList);
    vecLog(vid, `vec0 results: ${vec0Results.length}`);
    if (vec0Results.length > 0) return vec0Results;
    vecLog(vid, 'vec0 empty, trying scalar...');
  }

  if (capability === 'vec0' || capability === 'vec_scalar') {
    const scalarResults = await searchVecScalar(db, queryVector, limit, category, modelList);
    vecLog(vid, `scalar results: ${scalarResults.length}`);
    if (scalarResults.length > 0) return scalarResults;
  }

  vecLog(vid, 'falling back to JS cosine...');
  const jsResults = await searchJsCosine(db, queryVector, limit, category, modelList);
  vecLog(vid, `JS cosine results: ${jsResults.length}`);
  return jsResults;
}
