// src/pages/knowledge/hooks/useKnowledgeLibrary.ts
//
// Хук CRUD-операций с библиотекой документов.

import { useState, useCallback, useEffect } from 'react';
import { RagFacade } from '../../../services/rag/RagFacade';
import type { LibraryDocumentWithStatus } from '../../../services/rag/knowledgeLibrary/types';

export function useKnowledgeLibrary() {
  const [documents, setDocuments] = useState<LibraryDocumentWithStatus[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const docs = await RagFacade.getLibraryDocuments();
      setDocuments(docs);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const deleteDocument = useCallback(async (id: number) => {
    try {
      await RagFacade.deleteLibraryDocument(id);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
    } catch (e: any) {
      setError(e?.message ?? String(e));
    }
  }, []);

  const updateDocument = useCallback(
    async (id: number, data: { title?: string; category?: string }) => {
      try {
        await RagFacade.updateLibraryDocument(id, data);
        await load();
      } catch (e: any) {
        setError(e?.message ?? String(e));
      }
    },
    [load],
  );

  const categories = Array.from(new Set(documents.map((d) => d.category))).sort();

  useEffect(() => {
    load();
  }, [load]);

  return {
    documents,
    categories,
    loading,
    error,
    reload: load,
    deleteDocument,
    updateDocument,
  };
}
