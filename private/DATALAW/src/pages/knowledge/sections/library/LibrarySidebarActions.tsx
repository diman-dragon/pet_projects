import { Loader2, RefreshCw, Upload } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

export function LibrarySidebarActions({
  isUploading,
  isIndexing,
  onUpload,
  onRefresh,
}: {
  isUploading: boolean;
  isIndexing: boolean;
  onUpload: () => void;
  onRefresh: () => void;
}) {
  const { t } = useI18n();

  return (
    <div className="p-3 border-b border-black/[0.06] space-y-2">
      <button
        onClick={onUpload}
        disabled={isUploading}
        className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[var(--app-accent)] text-white text-xs font-medium rounded-lg hover:opacity-90 disabled:opacity-50 transition-opacity"
      >
        {isUploading ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            {t('knowledge.library.uploading')}
          </>
        ) : (
          <>
            <Upload className="w-3.5 h-3.5" />
            {t('knowledge.library.upload')}
          </>
        )}
      </button>

      <button
        onClick={onRefresh}
        disabled={isIndexing}
        className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs text-black/60 bg-black/[0.03] hover:bg-black/[0.06] rounded-lg transition-colors disabled:opacity-50"
      >
        <RefreshCw className={`w-3.5 h-3.5 ${isIndexing ? 'animate-spin' : ''}`} />
        {t('knowledge.library.refresh')}
      </button>
    </div>
  );
}
