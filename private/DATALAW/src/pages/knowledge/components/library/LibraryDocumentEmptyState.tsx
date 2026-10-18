import { FileText } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

export function LibraryDocumentEmptyState() {
  const { t } = useI18n();

  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="mb-3 flex size-12 items-center justify-center rounded-2xl bg-black/[0.04]">
        <FileText size={20} className="text-black/20" />
      </div>
      <p className="text-[length:var(--text-sm)] font-medium text-black/40">
        {t('knowledge.library.emptyTitle')}
      </p>
      <p className="mt-1 text-[10px] text-black/25">
        {t('knowledge.library.emptyHint')}{' '}
        <code className="rounded bg-black/[0.05] px-1">Knowledge_Base/</code>
      </p>
    </div>
  );
}
