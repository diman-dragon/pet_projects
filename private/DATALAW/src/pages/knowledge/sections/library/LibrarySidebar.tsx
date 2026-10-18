import { LibrarySidebarActions } from './LibrarySidebarActions';
import { LibrarySidebarFolders } from './LibrarySidebarFolders';
import { LibrarySidebarReindex } from './LibrarySidebarReindex';

interface LibrarySidebarProps {
  documentsCount: number;
  folders: string[];
  selectedFolder: string | null;
  folderCounts: Record<string, number>;
  isUploading: boolean;
  isIndexing: boolean;
  reindexProgress: { current: number; total: number; documentName: string } | null;
  onSelectFolder: (folder: string | null) => void;
  onUpload: () => void;
  onRefresh: () => void;
  onReindexAll: () => void;
}

export function LibrarySidebar(props: LibrarySidebarProps) {
  return (
    <div className="w-56 shrink-0 border-r border-black/[0.06] bg-white flex flex-col">
      <LibrarySidebarActions
        isUploading={props.isUploading}
        isIndexing={props.isIndexing}
        onUpload={props.onUpload}
        onRefresh={props.onRefresh}
      />
      <LibrarySidebarFolders
        documentsCount={props.documentsCount}
        folders={props.folders}
        selectedFolder={props.selectedFolder}
        folderCounts={props.folderCounts}
        onSelectFolder={props.onSelectFolder}
      />
      <LibrarySidebarReindex
        isIndexing={props.isIndexing}
        reindexProgress={props.reindexProgress}
        onReindexAll={props.onReindexAll}
      />
    </div>
  );
}
