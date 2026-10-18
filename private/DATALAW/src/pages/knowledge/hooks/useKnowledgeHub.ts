import { useState } from 'react';
import { useFullTextKnowledge } from './useFullTextKnowledge';
import { useKnowledgeLibraryHub } from './useKnowledgeLibraryHub';
import { useSemanticKnowledge } from './useSemanticKnowledge';
import type { KnowledgeTab } from './useKnowledgeHub.types';

export type { ChatMessage } from './useKnowledgeHub.types';

export function useKnowledgeHub() {
  const [activeTab, setActiveTab] = useState<KnowledgeTab>('semantic');
  const semantic = useSemanticKnowledge();
  const fullText = useFullTextKnowledge();
  const library = useKnowledgeLibraryHub();

  return {
    activeTab,
    setActiveTab,
    semanticState: semantic.semanticState,
    sendMessage: semantic.sendMessage,
    abortSemantic: semantic.abortSemantic,
    clearChat: semantic.clearChat,
    memory: semantic.memory,
    fullTextState: fullText.fullTextState,
    searchFullText: fullText.searchFullText,
    libraryState: library.libraryState,
    setLibraryState: library.setLibraryState,
    uploadDocuments: library.uploadDocuments,
    moveToFolder: library.moveToFolder,
    reindexDocument: library.reindexDocument,
    reindexAll: library.reindexAll,
    deleteDocument: library.deleteDocument,
    loadDocuments: library.loadDocuments,
  };
}
