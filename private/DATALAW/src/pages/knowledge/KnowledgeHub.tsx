import { useMemo, useState } from 'react';
import { Database } from 'lucide-react';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { PageBase } from '../../shared/ui/PageBase';
import { KnowledgeHubContent } from './KnowledgeHubContent';
import { buildKnowledgeHubTabs } from './KnowledgeHubTabs';
import { useKnowledgeHub } from './hooks/useKnowledgeHub';
import type { KnowledgeSourceSelection } from './types/source';

export function KnowledgeHub() {
  const { t } = useI18n();
  const {
    activeTab,
    setActiveTab,
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
    moveToFolder,
    reindexDocument,
    reindexAll,
    deleteDocument,
  } = useKnowledgeHub();
  const [selectedSource, setSelectedSource] = useState<KnowledgeSourceSelection | null>(null);

  const tabs = useMemo(() => buildKnowledgeHubTabs(t, libraryState.documents.length), [libraryState.documents.length, t]);

  return (
    <PageBase
      title={t('knowledge.hub.title')}
      tabs={tabs}
      activeTab={activeTab}
      onTabChange={(tab) => {
        setActiveTab(tab as 'semantic' | 'fulltext' | 'library' | 'indexing');
        if (tab !== 'semantic') {
          setSelectedSource(null);
        }
      }}
      icon={<Database size={20} className="text-[var(--app-accent)]" />}
      noPadding
    >
      <div className="h-full">
        <KnowledgeHubContent
          hub={{
            activeTab,
            setActiveTab,
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
            moveToFolder,
            reindexDocument,
            reindexAll,
            deleteDocument,
          }}
          selectedSource={selectedSource}
          onSelectSource={setSelectedSource}
          onResetSource={() => setSelectedSource(null)}
        />
      </div>
    </PageBase>
  );
}

export default KnowledgeHub;
