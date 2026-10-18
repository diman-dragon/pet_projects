import { IndexerPanel } from './components/indexer/IndexerPanel';
import { LibraryTab } from './sections/library/LibraryTab';
import { FullTextSearchTab } from './sections/fulltext/FullTextSearchTab';
import { SemanticSearchTab } from './sections/semantic/SemanticSearchTab';
import type { KnowledgeSourceSelection } from './types/source';
import type { KnowledgeHubViewModel } from './hooks/useKnowledgeHub.helpers';

export function KnowledgeHubContent({
  hub,
  selectedSource,
  onSelectSource,
  onResetSource,
}: {
  hub: KnowledgeHubViewModel;
  selectedSource: KnowledgeSourceSelection | null;
  onSelectSource: (source: KnowledgeSourceSelection | null) => void;
  onResetSource: () => void;
}) {
  const {
    activeTab,
    semanticState,
    sendMessage,
    abortSemantic,
    clearChat,
    fullTextState,
    searchFullText,
    libraryState,
    setLibraryState,
    uploadDocuments,
    loadDocuments,
    reindexDocument,
    reindexAll,
    deleteDocument,
    setActiveTab,
  } = hub;

  if (activeTab === 'semantic') {
    return (
      <SemanticSearchTab
        messages={semanticState.messages}
        isStreaming={semanticState.isStreaming}
        isThinking={semanticState.isThinking}
        onSendMessage={sendMessage}
        onAbort={abortSemantic}
        onClear={clearChat}
        onSelectSource={(source) => onSelectSource(source)}
        selectedSource={selectedSource}
      />
    );
  }

  if (activeTab === 'fulltext') {
    return (
      <FullTextSearchTab
        query={fullTextState.query}
        results={fullTextState.results}
        isSearching={fullTextState.isSearching}
        folders={libraryState.folders}
        selectedFolder={fullTextState.selectedFolder}
        onSearch={searchFullText}
      />
    );
  }

  if (activeTab === 'library') {
    return (
      <LibraryTab
        documents={libraryState.documents}
        folders={libraryState.folders}
        selectedFolder={libraryState.selectedFolder}
        isUploading={libraryState.isUploading}
        isIndexing={libraryState.isIndexing}
        reindexProgress={libraryState.reindexProgress}
        onSelectFolder={(folder) =>
          setLibraryState((prev) => ({ ...prev, selectedFolder: folder }))
        }
        onUpload={uploadDocuments}
        onRefresh={loadDocuments}
        onReindex={reindexDocument}
        onReindexAll={reindexAll}
        onDelete={deleteDocument}
      />
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <IndexerPanel
        onStartIndexing={() => {
          onResetSource();
          setActiveTab('indexing');
        }}
      />
    </div>
  );
}
