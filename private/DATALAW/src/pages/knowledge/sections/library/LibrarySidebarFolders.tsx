import { Database, Folder, FolderOpen } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

export function LibrarySidebarFolders({
  documentsCount,
  folders,
  selectedFolder,
  folderCounts,
  onSelectFolder,
}: {
  documentsCount: number;
  folders: string[];
  selectedFolder: string | null;
  folderCounts: Record<string, number>;
  onSelectFolder: (folder: string | null) => void;
}) {
  const { t } = useI18n();

  return (
    <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
      <button
        onClick={() => onSelectFolder(null)}
        className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-colors ${
          selectedFolder === null
            ? 'bg-[var(--app-accent)]/10 text-[var(--app-accent)] font-medium'
            : 'text-black/60 hover:bg-black/[0.03]'
        }`}
      >
        <Database className="w-3.5 h-3.5 shrink-0" />
        <span className="flex-1 text-left">{t('knowledge.library.allDocuments')}</span>
        <span className="text-[10px] text-black/40">{documentsCount}</span>
      </button>

      {folders.length > 0 ? (
        <div className="pt-2 pb-1 px-3">
          <p className="text-[9px] uppercase tracking-wider text-black/35 font-semibold">
            {t('knowledge.library.folders')}
          </p>
        </div>
      ) : null}

      {folders.map((folder) => (
        <button
          key={folder}
          onClick={() => onSelectFolder(folder)}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-colors ${
            selectedFolder === folder
              ? 'bg-[var(--app-accent)]/10 text-[var(--app-accent)] font-medium'
              : 'text-black/60 hover:bg-black/[0.03]'
          }`}
        >
          {selectedFolder === folder ? (
            <FolderOpen className="w-3.5 h-3.5 shrink-0" />
          ) : (
            <Folder className="w-3.5 h-3.5 shrink-0" />
          )}
          <span className="flex-1 text-left truncate">{folder}</span>
          <span className="text-[10px] text-black/40">{folderCounts[folder] || 0}</span>
        </button>
      ))}
    </div>
  );
}
