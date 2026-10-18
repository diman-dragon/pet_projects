import { FileText } from 'lucide-react';
import type { TranslationKey } from '../../../../i18n/translations';

export function FullTextResultsEmptyState({
  query,
  isSearching,
  t,
}: {
  query: string;
  isSearching: boolean;
  t: (key: TranslationKey, params?: Record<string, string | number>) => string;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center space-y-4 px-8 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-black/[0.03]">
        <FileText className="h-7 w-7 text-black/20" />
      </div>
      <div className="max-w-xs space-y-1.5">
        <p className="text-sm font-medium text-black/40">
          {query.trim() && !isSearching
            ? t('knowledge.fulltext.empty.noResultsTitle')
            : t('knowledge.fulltext.empty.idleTitle')}
        </p>
        <p className="text-xs leading-relaxed text-black/28">
          {query.trim() && !isSearching
            ? t('knowledge.fulltext.empty.noResultsHint')
            : t('knowledge.fulltext.empty.idleHint')}
        </p>
      </div>
    </div>
  );
}
