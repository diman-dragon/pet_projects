import { useCallback, useRef, useState } from 'react';
import { RagFacade } from '../../../services/rag/RagFacade';
import type { FullTextState } from './useKnowledgeHub.types';

export function useFullTextKnowledge() {
  const [fullTextState, setFullTextState] = useState<FullTextState>({
    query: '',
    results: [],
    isSearching: false,
    selectedFolder: null,
  });
  const requestIdRef = useRef(0);

  const searchFullText = useCallback(async (query: string, folder: string | null = null) => {
    const requestId = ++requestIdRef.current;

    if (!query.trim()) {
      setFullTextState((prev) => ({
        ...prev,
        results: [],
        query: '',
        selectedFolder: folder,
        isSearching: false,
      }));
      return;
    }

    setFullTextState((prev) => ({ ...prev, isSearching: true, query, selectedFolder: folder }));

    try {
      const results = await RagFacade.search({
        query,
        topK: 20,
        hybrid: false,
        category: folder || undefined,
      });

      if (requestId !== requestIdRef.current) {
        return;
      }

      setFullTextState((prev) => ({
        ...prev,
        results,
        isSearching: false,
      }));
    } catch (error) {
      console.error('[FULLTEXT] Error:', error);
      if (requestId !== requestIdRef.current) {
        return;
      }
      setFullTextState((prev) => ({
        ...prev,
        results: [],
        isSearching: false,
      }));
    }
  }, []);

  return {
    fullTextState,
    searchFullText,
  };
}
