import { Loader2 } from 'lucide-react';
import type { TranslationKey } from '../../../../i18n/translations';

export function FullTextSearchStatusBar({
  isSearching,
  filteredCount,
  hiddenByFilter,
  mainInput,
  query,
  t,
}: {
  isSearching: boolean;
  filteredCount: number;
  hiddenByFilter: number;
  mainInput: string;
  query: string;
  t: (key: TranslationKey, params?: Record<string, string | number>) => string;
}) {
  if (!mainInput && filteredCount === 0) {
    return null;
  }

  return (
    <div className="flex items-center gap-3 border-t border-black/[0.04] px-5 py-2">
      {isSearching ? (
        <span className="flex items-center gap-1.5 text-xs text-black/45">
          <Loader2 className="h-3 w-3 animate-spin" />
          {t('globalSearch.loading')}
        </span>
      ) : (
        <span className="text-xs text-black/45">
          {filteredCount > 0
            ? t('knowledge.fulltext.resultsFound', { count: filteredCount, query })
            : mainInput.trim()
              ? t('knowledge.fulltext.empty.noResultsTitle')
              : ''}
        </span>
      )}

      {hiddenByFilter > 0 ? (
        <span className="rounded-full bg-black/[0.04] px-2 py-0.5 text-[10px] text-black/38">
          {t('knowledge.fulltext.hiddenByFilters', { count: hiddenByFilter })}
        </span>
      ) : null}
    </div>
  );
}
