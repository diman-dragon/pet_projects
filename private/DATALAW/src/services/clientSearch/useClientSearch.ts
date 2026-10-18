// services/clientSearch/useClientSearch.ts
//
// Хук семантического поиска по клиентам и делам.
// Возвращает clientId / caseId для навигации.

import { useState, useCallback, useRef } from 'react';
import { RagFacade } from '../rag/RagFacade';
import type { SearchResult } from '../rag/UnifiedSearchService';

export interface ClientSearchResult {
  clientId: number | null;
  caseId: number | null;
  category: string;
  title: string;
  excerpt: string;
  score: number;
  filePath: string;
}

function toResult(r: SearchResult): ClientSearchResult {
  const clientMatch = r.filePath.match(/virtual:\/\/clients\/(\d+)/);
  const caseMatch = r.filePath.match(/virtual:\/\/cases\/(\d+)/);
  return {
    clientId: clientMatch ? Number(clientMatch[1]) : null,
    caseId: caseMatch ? Number(caseMatch[1]) : null,
    category: r.category,
    title: r.title,
    excerpt: r.chunkText.slice(0, 180).replace(/\n+/g, ' ').trim(),
    score: r.score,
    filePath: r.filePath,
  };
}

export function useClientSearch() {
  const [results, setResults] = useState<ClientSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [semanticAvailable, setSemanticAvailable] = useState(true);
  const abortRef = useRef<AbortController | null>(null);

  const search = useCallback(async (query: string) => {
    if (!query.trim()) { setResults([]); return; }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setSearching(true);

    try {
      const raw = await RagFacade.searchAll(query, 15);
      if (controller.signal.aborted) return;

      const virtual = raw
        .filter((r) => r.filePath.startsWith('virtual://'))
        .map(toResult);

      setResults(virtual);
      setSemanticAvailable(true);
    } catch {
      if (!controller.signal.aborted) {
        setResults([]);
        setSemanticAvailable(false);
      }
    } finally {
      if (!controller.signal.aborted) setSearching(false);
    }
  }, []);

  const clear = useCallback(() => {
    abortRef.current?.abort();
    setResults([]);
    setSearching(false);
  }, []);

  return { results, searching, semanticAvailable, search, clear };
}
