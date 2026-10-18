import { useMemo, useState } from 'react';
import type { LibraryDocumentWithStatus } from '../../../../services/rag/knowledgeLibrary/types';
import { LibraryDocumentList } from '../../components/library/LibraryDocumentList';
import { LibrarySearchBar } from './LibrarySearchBar';
import { LibrarySidebar } from './LibrarySidebar';
import { filterLibraryDocuments, getFolderCounts } from './libraryUtils';

interface LibraryTabProps {
  documents: LibraryDocumentWithStatus[];
  folders: string[];
  selectedFolder: string | null;
  isUploading: boolean;
  isIndexing: boolean;
  reindexProgress: { current: number; total: number; documentName: string } | null;
  onSelectFolder: (folder: string | null) => void;
  onUpload: () => void;
  onRefresh: () => void;
  onReindex: (documentId: number) => void;
  onReindexAll: () => void;
  onDelete: (documentId: number) => void;
}

export function LibraryTab({
  documents,
  folders,
  selectedFolder,
  isUploading,
  isIndexing,
  reindexProgress,
  onSelectFolder,
  onUpload,
  onRefresh,
  onReindex,
  onReindexAll,
  onDelete,
}: LibraryTabProps) {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredDocuments = useMemo(
    () => filterLibraryDocuments(documents, selectedFolder, searchQuery),
    [documents, searchQuery, selectedFolder],
  );
  const folderCounts = useMemo(() => getFolderCounts(documents, folders), [documents, folders]);
  const handleReindexAdapter = (_filePath: string, id: number) => onReindex(id);

  return (
    <div className="flex h-full overflow-hidden">
      <LibrarySidebar
        documentsCount={documents.length}
        folders={folders}
        selectedFolder={selectedFolder}
        folderCounts={folderCounts}
        isUploading={isUploading}
        isIndexing={isIndexing}
        reindexProgress={reindexProgress}
        onSelectFolder={onSelectFolder}
        onUpload={onUpload}
        onRefresh={onRefresh}
        onReindexAll={onReindexAll}
      />

      <div className="flex-1 flex flex-col min-w-0 bg-[var(--app-surface-soft)]">
        <LibrarySearchBar
          searchQuery={searchQuery}
          filteredCount={filteredDocuments.length}
          totalCount={documents.length}
          onChange={setSearchQuery}
        />

        <div className="flex-1 overflow-y-auto p-4">
          <LibraryDocumentList
            documents={filteredDocuments}
            loading={isIndexing || isUploading}
            onDelete={onDelete}
            onReindex={handleReindexAdapter}
          />
        </div>
      </div>
    </div>
  );
}
