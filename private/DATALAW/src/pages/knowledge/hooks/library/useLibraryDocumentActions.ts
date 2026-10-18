import { useCallback, type Dispatch, type SetStateAction } from 'react';
import { KnowledgeLibraryService } from '../../../../services/rag/KnowledgeLibraryService';
import type { LibraryState } from '../useKnowledgeHub.types';
import { uploadKnowledgeDocuments } from './uploadDocuments';

interface UseLibraryDocumentActionsParams {
  setLibraryState: Dispatch<SetStateAction<LibraryState>>;
}

export function useLibraryDocumentActions({ setLibraryState }: UseLibraryDocumentActionsParams) {
  const loadDocuments = useCallback(async () => {
    setLibraryState((prev) => ({ ...prev, isIndexing: true }));
    try {
      await KnowledgeLibraryService.scanAndAddFolder();
      const documents = await KnowledgeLibraryService.getAll();
      setLibraryState((prev) => ({ ...prev, documents, isIndexing: false }));
    } catch (error) {
      console.error('[LIBRARY] Failed to load documents:', error);
      setLibraryState((prev) => ({ ...prev, isIndexing: false }));
    }
  }, [setLibraryState]);

  const uploadDocuments = useCallback(async () => {
    setLibraryState((prev) => ({ ...prev, isUploading: true }));
    try {
      const uploaded = await uploadKnowledgeDocuments();
      if (uploaded) {
        await loadDocuments();
      }
    } catch (error) {
      console.error('[LIBRARY] Upload error:', error);
    } finally {
      setLibraryState((prev) => ({ ...prev, isUploading: false }));
    }
  }, [loadDocuments, setLibraryState]);

  const moveToFolder = useCallback(
    async (documentId: number, folder: string) => {
      try {
        await KnowledgeLibraryService.update(documentId, { category: folder });
        await loadDocuments();
      } catch (error) {
        console.error('[LIBRARY] Move error:', error);
      }
    },
    [loadDocuments],
  );

  const deleteDocument = useCallback(
    async (documentId: number) => {
      try {
        await KnowledgeLibraryService.delete(documentId);
        await loadDocuments();
      } catch (error) {
        console.error('[LIBRARY] Delete error:', error);
      }
    },
    [loadDocuments],
  );

  return {
    loadDocuments,
    uploadDocuments,
    moveToFolder,
    deleteDocument,
  };
}
