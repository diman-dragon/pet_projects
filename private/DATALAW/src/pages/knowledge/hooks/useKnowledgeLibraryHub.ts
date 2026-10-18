import { useEffect, useState } from 'react';
import { createInitialLibraryState } from './library/libraryHubState';
import { useEmbeddingModelWatcher } from './library/useEmbeddingModelWatcher';
import { useLibraryHubActions } from './library/useLibraryHubActions';

export function useKnowledgeLibraryHub() {
  const [libraryState, setLibraryState] = useState(createInitialLibraryState);
  const {
    loadDocuments,
    uploadDocuments,
    moveToFolder,
    reindexDocument,
    reindexAll,
    deleteDocument,
  } = useLibraryHubActions({
    documents: libraryState.documents,
    setLibraryState,
  });

  useEmbeddingModelWatcher(reindexAll);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments]);

  return {
    libraryState,
    setLibraryState,
    loadDocuments,
    uploadDocuments,
    moveToFolder,
    reindexDocument,
    reindexAll,
    deleteDocument,
  };
}
