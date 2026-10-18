import { useCallback, useEffect, useState } from 'react';
import { AppPaths } from '../../../services/config/AppPaths';
import { useIndexerStore } from '../../../store/indexerStore';
import { useFavorites } from './useFavorites';
import { useKnowledgeLibrary } from './useKnowledgeLibrary';
import { useSearchPipeline } from './useSearchPipeline';

export type KnowledgeTab = 'search' | 'library';

export function useKnowledgeLogic() {
  const [activeTab, setActiveTab] = useState<KnowledgeTab>('search');
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState<string>('all');

  const pipeline = useSearchPipeline();
  const { favorites, toggleFavorite, isFavorite } = useFavorites();
  const library = useKnowledgeLibrary();
  const indexerStore = useIndexerStore();
  const libraryReload = library.reload;

  useEffect(() => {
    indexerStore.loadStats();
    libraryReload();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSearch = useCallback(() => {
    const words = query.trim().split(/\s+/).filter(Boolean);
    if (words.length < 2) return;
    pipeline.runSearch(query, category !== 'all' ? category : undefined);
  }, [query, category, pipeline.runSearch]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch();
  }, [handleSearch]);

  const clearSearch = useCallback(() => {
    setQuery('');
    setCategory('all');
    pipeline.reset();
  }, [pipeline.reset]);

  const startFolderIndexing = useCallback(async () => {
    const rootPath = AppPaths.getRootPath();
    if (!rootPath) return;
    const { join } = await import('@tauri-apps/api/path');
    const kbPath = await join(rootPath, 'Knowledge_Base');
    await indexerStore.startIndexing(kbPath);
    await libraryReload();
  }, [indexerStore, libraryReload]);

  const startFolderScan = useCallback(async () => {
    const rootPath = AppPaths.getRootPath();
    if (!rootPath) return;
    const { join } = await import('@tauri-apps/api/path');
    const kbPath = await join(rootPath, 'Knowledge_Base');
    await indexerStore.startScan(kbPath);
    await libraryReload();
  }, [indexerStore, libraryReload]);

  return {
    activeTab, setActiveTab,
    query, setQuery,
    category, setCategory,
    handleSearch, handleKeyDown,
    clearSearch,
    pipeline,
    library,
    indexer: {
      status: indexerStore.status,
      progress: indexerStore.progress,
      error: indexerStore.error,
      stats: indexerStore.stats,
      abort: indexerStore.abort,
      reset: indexerStore.reset,
      loadStats: indexerStore.loadStats,
      startIndexing: indexerStore.startIndexing,
      indexSingleFile: indexerStore.indexSingleFile,
      scan: indexerStore.startScan,
    },
    startFolderIndexing, startFolderScan,
    favorites, toggleFavorite, isFavorite,
  };
}
