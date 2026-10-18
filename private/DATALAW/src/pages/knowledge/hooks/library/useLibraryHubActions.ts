import type { Dispatch, SetStateAction } from 'react';
import type { LibraryState } from '../useKnowledgeHub.types';
import { useLibraryDocumentActions } from './useLibraryDocumentActions';
import { useLibraryReindexActions } from './useLibraryReindexActions';

interface UseLibraryHubActionsParams {
  documents: LibraryState['documents'];
  setLibraryState: Dispatch<SetStateAction<LibraryState>>;
}

export function useLibraryHubActions({
  documents,
  setLibraryState,
}: UseLibraryHubActionsParams) {
  const documentActions = useLibraryDocumentActions({ setLibraryState });
  const reindexActions = useLibraryReindexActions({
    documents,
    setLibraryState,
    loadDocuments: documentActions.loadDocuments,
  });

  return {
    ...documentActions,
    ...reindexActions,
  };
}
