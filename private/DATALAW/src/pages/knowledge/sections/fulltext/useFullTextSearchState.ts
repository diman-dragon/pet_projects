import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useDebounce } from '../../hooks/useDebounce';
import { DEFAULT_FILTERS } from './constants';
import type { FilterState, FullTextSearchTabProps } from './types';

export function useFullTextSearchState({
  query,
  results,
  selectedFolder,
  onSearch,
}: Pick<FullTextSearchTabProps, 'query' | 'results' | 'selectedFolder' | 'onSearch'>) {
  const [mainInput, setMainInput] = useState(query);
  const [filters, setFilters] = useState<FilterState>({
    ...DEFAULT_FILTERS,
    folder: selectedFolder ?? '',
  });
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [selectedChunkId, setSelectedChunkId] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fragmentRef = useRef<HTMLDivElement>(null);
  const [debouncedInput] = useDebounce(mainInput, 400);
  const [debouncedFolder] = useDebounce(filters.folder, 400);

  useEffect(() => {
    setMainInput(query);
  }, [query]);

  useEffect(() => {
    setFilters((prev) => ({ ...prev, folder: selectedFolder ?? prev.folder }));
  }, [selectedFolder]);

  useEffect(() => {
    if (debouncedInput.trim()) {
      onSearch(debouncedInput.trim(), debouncedFolder || null);
      return;
    }
    onSearch('', debouncedFolder || null);
  }, [debouncedFolder, debouncedInput, onSearch]);

  const availableCategories = useMemo(
    () => Array.from(new Set(results.map((result) => result.category).filter(Boolean))).sort(),
    [results],
  );
  const availableLanguages = useMemo(
    () => Array.from(new Set(results.map((result) => result.lang).filter(Boolean) as string[])).sort(),
    [results],
  );

  const filteredResults = useMemo(() => {
    let next = [...results];
    if (filters.titleQuery.trim()) {
      const titleQuery = filters.titleQuery.trim().toLowerCase();
      next = next.filter((result) => result.title.toLowerCase().includes(titleQuery));
    }
    if (filters.category) next = next.filter((result) => result.category === filters.category);
    if (filters.language) next = next.filter((result) => result.lang === filters.language);
    if (filters.folder) {
      next = next.filter((result) => result.filePath.toLowerCase().includes(filters.folder.toLowerCase()));
    }
    if (filters.fileType) {
      next = next.filter((result) => result.filePath.toLowerCase().endsWith(`.${filters.fileType}`));
    }
    if (filters.matchType !== 'all') next = next.filter((result) => result.matchType === filters.matchType);
    if (filters.minScore > 0) next = next.filter((result) => result.score >= filters.minScore / 100);
    return next.slice(0, filters.topK);
  }, [filters, results]);

  const selectedResult = useMemo(
    () => filteredResults.find((result) => result.chunkId === selectedChunkId) ?? null,
    [filteredResults, selectedChunkId],
  );

  useEffect(() => {
    if (selectedChunkId && !selectedResult) {
      setSelectedChunkId(null);
    }
  }, [selectedChunkId, selectedResult]);

  const hasActiveFilters = Boolean(
    filters.titleQuery ||
      filters.category ||
      filters.language ||
      filters.folder ||
      filters.fileType ||
      filters.matchType !== 'all' ||
      filters.minScore > 0,
  );
  const highlightTerms = useMemo(() => [mainInput, filters.titleQuery].filter(Boolean), [filters.titleQuery, mainInput]);
  const hiddenByFilter = Math.max(results.length - filteredResults.length, 0);
  const selectedIndex = selectedResult
    ? filteredResults.findIndex((result) => result.chunkId === selectedResult.chunkId)
    : -1;

  const handleFilterChange = useCallback((patch: Partial<FilterState>) => {
    setFilters((prev) => ({ ...prev, ...patch }));
  }, []);

  const handleSelectResult = useCallback((chunkId: number) => {
    setSelectedChunkId((prev) => (prev === chunkId ? null : chunkId));
    setTimeout(() => fragmentRef.current?.scrollTo({ top: 0, behavior: 'smooth' }), 40);
  }, []);

  return {
    mainInput,
    setMainInput,
    filters,
    setFilters,
    filtersOpen,
    setFiltersOpen,
    inputRef,
    fragmentRef,
    availableCategories,
    availableLanguages,
    filteredResults,
    selectedChunkId,
    setSelectedChunkId,
    selectedResult,
    hasActiveFilters,
    highlightTerms,
    hiddenByFilter,
    selectedIndex,
    handleFilterChange,
    handleSelectResult,
  };
}
