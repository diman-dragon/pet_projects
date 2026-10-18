import { useCallback } from 'react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { ShellService } from '../../../../services/shell/ShellService';
import { FullTextFiltersPanel } from './FullTextFiltersPanel';
import { FullTextFragmentPanel } from './FullTextFragmentPanel';
import { FullTextResultsPane } from './FullTextResultsPane';
import { FullTextSearchHeader } from './FullTextSearchHeader';
import { FullTextSearchStatusBar } from './FullTextSearchStatusBar';
import { DEFAULT_FILTERS } from './constants';
import { fullTextStyles } from './fullTextStyles';
import type { FullTextSearchTabProps } from './types';
import { useFullTextSearchState } from './useFullTextSearchState';

export function FullTextSearchTab(props: FullTextSearchTabProps) {
  const { t } = useI18n();
  const { query, isSearching, folders } = props;
  const state = useFullTextSearchState(props);

  const handleOpenDocument = useCallback(async (filePath: string) => {
    try {
      await ShellService.open(filePath);
    } catch (error) {
      console.error('[FTS] Cannot open document:', error);
    }
  }, []);

  return (
    <>
      <style>{fullTextStyles}</style>
      <div className="flex h-full flex-col overflow-hidden bg-[var(--app-surface)]">
        <div className="shrink-0 border-b border-black/[0.06] bg-white/95">
          <FullTextSearchHeader
            value={state.mainInput}
            isSearching={isSearching}
            folders={folders}
            selectedFolder={state.filters.folder}
            filtersOpen={state.filtersOpen}
            hasActiveFilters={state.hasActiveFilters}
            inputRef={state.inputRef}
            onChange={state.setMainInput}
            onFolderChange={(folder) => state.handleFilterChange({ folder })}
            onToggleFilters={() => state.setFiltersOpen((prev) => !prev)}
            t={t}
          />

          {state.filtersOpen ? (
            <FullTextFiltersPanel
              filters={state.filters}
              folders={folders}
              categories={state.availableCategories}
              languages={state.availableLanguages}
              hasActiveFilters={state.hasActiveFilters}
              onChange={state.handleFilterChange}
              onReset={() => state.setFilters({ ...DEFAULT_FILTERS })}
              t={t}
            />
          ) : null}

          <FullTextSearchStatusBar
            isSearching={isSearching}
            filteredCount={state.filteredResults.length}
            hiddenByFilter={state.hiddenByFilter}
            mainInput={state.mainInput}
            query={query}
            t={t}
          />
        </div>

        <div className="flex min-h-0 flex-1 overflow-hidden">
          <div className="min-w-0 flex-1 overflow-y-auto">
            <FullTextResultsPane
              results={state.filteredResults}
              isSearching={isSearching}
              query={state.mainInput}
              selectedChunkId={state.selectedChunkId ?? null}
              highlightTerms={state.highlightTerms}
              onSelect={(result) => state.handleSelectResult(result.chunkId)}
              t={t}
            />
          </div>

          <FullTextFragmentPanel
            result={state.selectedResult}
            selectedIndex={state.selectedIndex}
            highlightTerms={state.highlightTerms}
            fragmentRef={state.fragmentRef}
            onClose={() => state.setSelectedChunkId(null)}
            onOpenDocument={handleOpenDocument}
            t={t}
          />
        </div>
      </div>
    </>
  );
}
