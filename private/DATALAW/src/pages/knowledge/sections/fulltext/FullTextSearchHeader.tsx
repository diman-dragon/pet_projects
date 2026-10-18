// UPDATED: история через HistoryPanel (portal) + кнопка-триггер
import { ChevronDown, ChevronUp, Loader2, Search, SlidersHorizontal, X } from 'lucide-react';
import type { TranslationKey } from '../../../../i18n/translations';
import { useSearchHistory } from '../../../../hooks/useSearchHistory';
import { HistoryPanel, HistoryButton, useHistoryPanel } from '../../../../components/common/HistoryPanel';

interface FullTextSearchHeaderProps {
  value: string;
  isSearching: boolean;
  folders: string[];
  selectedFolder: string;
  filtersOpen: boolean;
  hasActiveFilters: boolean;
  inputRef: React.RefObject<HTMLInputElement | null>;
  onChange: (value: string) => void;
  onFolderChange: (value: string) => void;
  onToggleFilters: () => void;
  t: (key: TranslationKey) => string;
}

export function FullTextSearchHeader({
  value,
  isSearching,
  folders,
  selectedFolder,
  filtersOpen,
  hasActiveFilters,
  inputRef,
  onChange,
  onFolderChange,
  onToggleFilters,
  t,
}: FullTextSearchHeaderProps) {
  const hp = useHistoryPanel();
  const { history, deleteOne, clearAll } = useSearchHistory(hp.open);

  const historyItems = history.map((h) => ({ key: h.query, text: h.query }));

  return (
    <div className="flex items-center gap-3 px-5 py-4">
      <div className="relative flex-1">
        {isSearching ? (
          <Loader2 className="pointer-events-none absolute left-4 top-1/2 h-[18px] w-[18px] -translate-y-1/2 animate-spin text-[var(--app-accent)]" />
        ) : (
          <Search className="pointer-events-none absolute left-4 top-1/2 h-[18px] w-[18px] -translate-y-1/2 text-black/30" />
        )}
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={t('knowledge.fulltext.searchPlaceholder')}
          className="w-full rounded-xl border border-black/[0.07] bg-[var(--app-surface-soft)] py-3 pl-11 pr-20 text-sm outline-none transition-all placeholder:text-black/25 focus:border-[var(--app-accent)]/40 focus:bg-white focus:ring-2 focus:ring-[var(--app-accent)]/10"
          autoFocus
        />
        {/* Кнопки справа: история + очистить */}
        <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
          <HistoryButton
            open={hp.open}
            onClick={hp.toggle}
            btnRef={hp.btnRef}
            hasItems={historyItems.length > 0}
            className="p-1.5"
          />
          {value && (
            <button
              type="button"
              onClick={() => { onChange(''); inputRef.current?.focus(); }}
              className="p-1.5 rounded-xl hover:bg-black/[0.05] transition-colors"
            >
              <X className="h-3.5 w-3.5 text-black/35" />
            </button>
          )}
        </div>

        <HistoryPanel
          items={historyItems}
          onSelect={(text) => { onChange(text); inputRef.current?.focus(); }}
          onDeleteOne={deleteOne}
          onClearAll={clearAll}
          label="Недавние запросы"
          anchorRef={hp.btnRef as React.RefObject<HTMLElement | null>}
          open={hp.open}
          onClose={hp.close}
        />
      </div>

      {folders.length > 0 && (
        <select
          value={selectedFolder}
          onChange={(e) => onFolderChange(e.target.value)}
          className="min-w-[150px] rounded-xl border border-black/[0.07] bg-[var(--app-surface-soft)] px-3 py-3 text-xs text-black/60 outline-none"
        >
          <option value="">{t('knowledge.fulltext.folders.all')}</option>
          {folders.map((folder) => (
            <option key={folder} value={folder}>{folder}</option>
          ))}
        </select>
      )}

      <button
        type="button"
        onClick={onToggleFilters}
        className={`flex items-center gap-1.5 rounded-xl border p-3 transition-colors ${
          filtersOpen || hasActiveFilters
            ? 'border-[var(--app-accent)]/25 bg-[var(--app-accent)]/8 text-[var(--app-accent)]'
            : 'border-black/[0.07] bg-[var(--app-surface-soft)] text-black/45 hover:text-black/65'
        }`}
      >
        <SlidersHorizontal className="h-4 w-4" />
        {hasActiveFilters && <span className="h-1.5 w-1.5 rounded-full bg-[var(--app-accent)]" />}
        {filtersOpen ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
      </button>
    </div>
  );
}
