import { Loader2, RefreshCw } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

export function LibrarySidebarReindex({
  isIndexing,
  reindexProgress,
  onReindexAll,
}: {
  isIndexing: boolean;
  reindexProgress: { current: number; total: number; documentName: string } | null;
  onReindexAll: () => void;
}) {
  const { t } = useI18n();

  return (
    <div className="p-3 border-t border-black/[0.06]">
      {isIndexing && reindexProgress ? (
        <div className="mb-2">
          <div className="h-1.5 w-full bg-black/[0.08] rounded-full overflow-hidden mb-1.5">
            <div
              className="h-full bg-[var(--app-accent)] rounded-full transition-all duration-300"
              style={{ width: `${(reindexProgress.current / reindexProgress.total) * 100}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-[9px] text-black/50">
            <span className="truncate flex-1 mr-2" title={reindexProgress.documentName}>
              {reindexProgress.documentName}
            </span>
            <span className="shrink-0 font-medium">
              {reindexProgress.current} / {reindexProgress.total}
            </span>
          </div>
        </div>
      ) : null}

      <button
        onClick={onReindexAll}
        disabled={isIndexing}
        className="w-full flex items-center justify-center gap-2 px-3 py-2 text-[11px] text-black/50 hover:text-[var(--app-accent)] hover:bg-[var(--app-accent)]/5 rounded-lg transition-colors disabled:opacity-50"
      >
        {isIndexing ? (
          <>
            <Loader2 className="w-3 h-3 animate-spin" />
            {t('knowledge.library.reindexing')}
          </>
        ) : (
          <>
            <RefreshCw className="w-3 h-3" />
            {t('knowledge.library.reindexAll')}
          </>
        )}
      </button>
    </div>
  );
}
