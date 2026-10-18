// src/pages/knowledge/components/search/SearchBar.tsx

import { Jurisdiction, DocStatus } from '../../types';
import { SearchInput } from './SearchInput';
import { JurisdictionPills } from './JurisdictionPills';
import { FiltersToggle } from './FiltersToggle';
import { StatusFilter } from './StatusFilter';

interface SearchBarProps {
  query: string;
  onQueryChange: (v: string) => void;
  isSearching: boolean;
  jurisdiction: Jurisdiction;
  onJurisdictionChange: (v: Jurisdiction) => void;
  status: DocStatus;
  onStatusChange: (v: DocStatus) => void;
  showFilters: boolean;
  onToggleFilters: () => void;
  onClear: () => void;
}

/**
 * Панель поиска: строка ввода → пилюли юрисдикций → (опционально) фильтр статуса.
 */
export function SearchBar({
  query,
  onQueryChange,
  isSearching,
  jurisdiction,
  onJurisdictionChange,
  status,
  onStatusChange,
  showFilters,
  onToggleFilters,
  onClear,
}: SearchBarProps) {
  return (
    <div className="p-3 border-b border-black/[0.05] space-y-2">
      <SearchInput
        value={query}
        onChange={onQueryChange}
        onClear={onClear}
        onSearch={() => { }}
        onKeyDown={() => { }}
        isRunning={isSearching}
      />

      <div className="flex items-center justify-between gap-2">
        <JurisdictionPills value={jurisdiction} onChange={onJurisdictionChange} />
        <FiltersToggle isOpen={showFilters} onToggle={onToggleFilters} />
      </div>

      {showFilters && <StatusFilter value={status} onChange={onStatusChange} />}
    </div>
  );
}
