import { Search } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

export function LibrarySearchBar({
  searchQuery,
  filteredCount,
  totalCount,
  onChange,
}: {
  searchQuery: string;
  filteredCount: number;
  totalCount: number;
  onChange: (value: string) => void;
}) {
  const { t } = useI18n();

  return (
    <div className="px-4 py-3 border-b border-black/[0.06] bg-white flex items-center gap-3">
      <div className="flex-1 relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-black/35" />
        <input
          type="text"
          value={searchQuery}
          onChange={(event) => onChange(event.target.value)}
          placeholder={t('knowledge.library.searchPlaceholder')}
          className="w-full pl-9 pr-4 py-2 text-xs bg-black/[0.03] border-none rounded-lg outline-none focus:ring-2 focus:ring-[var(--app-accent)]/20"
        />
      </div>

      {searchQuery ? (
        <span className="text-[11px] text-black/40 whitespace-nowrap">
          {t('knowledge.library.filteredCount', { filtered: filteredCount, total: totalCount })}
        </span>
      ) : null}
    </div>
  );
}
