import { useCallback, type Dispatch, type SetStateAction } from 'react';
import { IndexerService } from '../../../../services/rag/IndexerService';
import { RagFacade } from '../../../../services/rag/RagFacade';
import type { LibraryState } from '../useKnowledgeHub.types';

interface UseLibraryReindexActionsParams {
  documents: LibraryState['documents'];
  setLibraryState: Dispatch<SetStateAction<LibraryState>>;
  loadDocuments: () => Promise<void>;
}

export function useLibraryReindexActions({
  documents,
  setLibraryState,
  loadDocuments,
}: UseLibraryReindexActionsParams) {
  const reindexDocument = useCallback(
    async (documentId: number) => {
      setLibraryState((prev) => ({ ...prev, isIndexing: true }));
      try {
        const document = documents.find((item) => item.id === documentId);
        if (document?.file_path) {
          await IndexerService.indexFile({
            filePath: document.file_path,
            title: document.title,
            category: document.category,
          });
        }
        await loadDocuments();
      } catch (error) {
        console.error('[LIBRARY] Reindex error:', error);
      } finally {
        setLibraryState((prev) => ({ ...prev, isIndexing: false }));
      }
    },
    [documents, loadDocuments, setLibraryState],
  );

  const reindexAll = useCallback(async () => {
    setLibraryState((prev) => ({
      ...prev,
      isIndexing: true,
      reindexProgress: { current: 0, total: prev.documents.length, documentName: '' },
    }));
    try {
      await RagFacade.forceReembedAll((progress) => {
        setLibraryState((prev) => ({
          ...prev,
          reindexProgress: {
            current: progress.done,
            total: progress.total,
            documentName: progress.currentTitle || '',
          },
        }));
      });
      await loadDocuments();
    } catch (error) {
      console.error('[LIBRARY] Mass reindex error:', error);
    } finally {
      setLibraryState((prev) => ({ ...prev, isIndexing: false, reindexProgress: null }));
    }
  }, [loadDocuments, setLibraryState]);

  return {
    reindexDocument,
    reindexAll,
  };
}
