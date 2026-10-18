// src/components/layout/Header/GlobalSearchModal.tsx
//
// ИТЕРАЦИЯ 4: SearchSuggestionsService, keyboard navigation (↑↓ Enter Escape),
// breadcrumbs в результатах, двухфазная выдача (FTS → AI placeholder),
// интеграция SearchCache.

import {
  useEffect,
  useMemo,
  useState,
  useRef,
  useCallback,
  type KeyboardEvent,
} from 'react';
import { Clock, User, FileText, Trash2, X } from 'lucide-react';
import { SearchInput } from '../../ui/form/SearchInput';
import { EmptyStatePanel } from '../../ui/atoms/EmptyStatePanel';
import { Modal } from '../../ui/overlays/Modal';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';
import {
  GlobalSearchService,
  type GlobalSearchResult,
} from '../../../services/search/GlobalSearchService';
import {
  SearchSuggestionsService,
  type SearchSuggestion,
  invalidateSuggestionsCache,
} from '../../../services/search/SearchSuggestionsService';
import { ConnectionManager } from '../../../services/db/ConnectionManager';
import { SearchCache } from '../../../services/search/SearchCache';
import type { UserProfile } from '../../../types/entities';
import { GlobalSearchResultRow } from './global-search/GlobalSearchResultRow';
import { GlobalSearchResultsSection } from './global-search/GlobalSearchResultsSection';

interface GlobalSearchModalProps {
  open: boolean;
  onClose: () => void;
  user: UserProfile;
  onNavigate: (route: string, params?: Record<string, unknown>) => void;
}

async function deleteHistoryItem(query: string): Promise<void> {
  if (!ConnectionManager.exists('kb_main')) return;
  try {
    const db = ConnectionManager.get('kb_main');
    await db.execute('DELETE FROM search_history WHERE query = $1', [query]);
    invalidateSuggestionsCache();
  } catch { /* ignore */ }
}

async function clearAllHistory(): Promise<void> {
  if (!ConnectionManager.exists('kb_main')) return;
  try {
    const db = ConnectionManager.get('kb_main');
    await db.execute('DELETE FROM search_history');
    invalidateSuggestionsCache();
  } catch { /* ignore */ }
}

function sectionLabelKey(section: GlobalSearchResult['section']): TranslationKey {
  return `globalSearch.section.${section}` as TranslationKey;
}

const SEARCH_DEBOUNCE_MS = 200;
const SUGGEST_DEBOUNCE_MS = 120;

// ── Keyboard navigation hook ──────────────────────────────────────────────────

function useFlatList(
  sections: [GlobalSearchResult['section'], GlobalSearchResult[]][],
): GlobalSearchResult[] {
  return useMemo(() => sections.flatMap(([, items]) => items), [sections]);
}

export function GlobalSearchModal({ open, onClose, user, onNavigate }: GlobalSearchModalProps) {
  const { t } = useI18n();
  const [query, setQuery] = useState('');

  // FTS results
  const [ftsResults, setFtsResults] = useState<GlobalSearchResult[]>([]);
  const [ftsLoading, setFtsLoading] = useState(false);

  // AI phase placeholder
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [aiLoading, setAiLoading] = useState(false);

  // Suggestions (пустой ввод → recent, иначе prefix)
  const [suggestions, setSuggestions] = useState<SearchSuggestion[]>([]);
  const [suggestMode, setSuggestMode] = useState(false);

  // Keyboard selection
  const [activeIdx, setActiveIdx] = useState(-1);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const suggestDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const reset = useCallback(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (suggestDebounceRef.current) clearTimeout(suggestDebounceRef.current);
    abortRef.current?.abort();
    setFtsResults([]);
    setFtsLoading(false);
    setAiAnswer(null);
    setAiLoading(false);
    setSuggestions([]);
    setSuggestMode(false);
    setActiveIdx(-1);
  }, []);

  // Reset on close
  useEffect(() => {
    if (!open) {
      setQuery('');
      reset();
    }
  }, [open, reset]);

  // Load recent activity on open with empty query
  useEffect(() => {
    if (open && !query.trim()) {
      SearchSuggestionsService.getRecentActivity().then(setSuggestions).catch(() => setSuggestions([]));
      setSuggestMode(true);
    }
  }, [open, query]);

  // Search pipeline
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    abortRef.current?.abort();
    setActiveIdx(-1);

    if (!open || !query.trim()) {
      reset();
      // Show recent activity if open
      if (open) {
        SearchSuggestionsService.getRecentActivity().then(setSuggestions).catch(() => setSuggestions([]));
        setSuggestMode(true);
      }
      return;
    }

    // Suggestions while typing
    if (suggestDebounceRef.current) clearTimeout(suggestDebounceRef.current);
    suggestDebounceRef.current = setTimeout(() => {
      if (query.trim().length >= 2) {
        SearchSuggestionsService.getSuggestions(query).then(setSuggestions).catch(() => setSuggestions([]));
        setSuggestMode(true);
      } else {
        setSuggestions([]);
        setSuggestMode(false);
      }
    }, SUGGEST_DEBOUNCE_MS);

    // Check cache before searching
    const cached = SearchCache.get(query.trim());
    if (cached) {
      setFtsResults(cached);
      setFtsLoading(false);
      setSuggestMode(false);
      return;
    }

    setFtsLoading(true);
    setAiAnswer(null);

    const controller = new AbortController();
    abortRef.current = controller;

    debounceRef.current = setTimeout(() => {
      GlobalSearchService.search(query, user, controller.signal)
        .then((items) => {
          if (controller.signal.aborted) return;
          setFtsResults(items);
          setFtsLoading(false);
          setSuggestMode(false);

          // Cache result
          SearchCache.set(query.trim(), query.trim().toLowerCase(), items);

          // AI phase placeholder (место зарезервировано)
          if (items.length > 0) {
            setAiLoading(true);
            setAiLoading(false);
          }
        })
        .catch(() => {
          if (!controller.signal.aborted) {
            setFtsResults([]);
            setFtsLoading(false);
          }
        });
    }, SEARCH_DEBOUNCE_MS);

    return () => {
      controller.abort();
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [open, query, user, reset]);

  const grouped = useMemo(() => {
    const buckets = new Map<GlobalSearchResult['section'], GlobalSearchResult[]>();
    for (const item of ftsResults) {
      buckets.set(item.section, [...(buckets.get(item.section) ?? []), item]);
    }
    return [...buckets.entries()];
  }, [ftsResults]);

  const flatList = useFlatList(grouped);
  const hasResults = ftsResults.length > 0;

  // ── Keyboard navigation ────────────────────────────────────────────────────

  const handlePick = useCallback(
    (item: GlobalSearchResult) => {
      GlobalSearchService.recordSelection(item.id).catch(() => {});
      onNavigate(item.target.route, item.target);
      onClose();
    },
    [onNavigate, onClose],
  );

  const handleSuggestionPick = useCallback(
    (suggestion: SearchSuggestion) => {
      setQuery(suggestion.text);
      setSuggestions([]);
      setSuggestMode(false);
    },
    [],
  );

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      // Items: suggestions OR flat results
      const items = suggestMode && suggestions.length > 0 ? suggestions : flatList;
      const total = items.length;

      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
        return;
      }

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveIdx((prev) => (prev < total - 1 ? prev + 1 : 0));
        return;
      }

      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveIdx((prev) => (prev > 0 ? prev - 1 : total - 1));
        return;
      }

      if (e.key === 'Enter' && activeIdx >= 0) {
        e.preventDefault();
        if (suggestMode && suggestions.length > 0 && suggestions[activeIdx]) {
          handleSuggestionPick(suggestions[activeIdx]);
        } else if (flatList[activeIdx]) {
          handlePick(flatList[activeIdx]);
        }
        return;
      }

      if (e.key === 'Tab') {
        // Tab между секциями — сбрасываем активный индекс
        setActiveIdx(-1);
      }
    },
    [suggestions, flatList, suggestMode, activeIdx, onClose, handlePick, handleSuggestionPick],
  );

  // Scroll active item into view
  useEffect(() => {
    if (activeIdx < 0 || !listRef.current) return;
    const el = listRef.current.querySelector(`[data-idx="${activeIdx}"]`) as HTMLElement | null;
    el?.scrollIntoView({ block: 'nearest' });
  }, [activeIdx]);

  return (
    <Modal open={open} onClose={onClose} title={t('globalSearch.title')} size="lg">
      <div className="space-y-4" onKeyDown={handleKeyDown}>
        <SearchInput
          autoFocus
          value={query}
          onChange={setQuery}
          placeholder={t('globalSearch.inputPlaceholder')}
          className="h-11 max-w-none rounded-[var(--app-radius-md)] text-[length:var(--text-sm)]"
        />

        {/* История поиска — встроенная в модалку (не portal, т.к. Modal уже поверх) */}
        {suggestMode && !hasResults && (
          <div className="space-y-1">
            <div className="flex items-center justify-between px-1 pb-1 border-b border-black/[0.05]">
              <span className="flex items-center gap-1.5 text-[11px] font-semibold text-black/35 uppercase tracking-widest">
                <Clock size={11} />
                {query.trim() ? 'Похожие запросы' : 'Недавняя активность'}
              </span>
              {suggestions.length > 0 && (
                <button
                  type="button"
                  onClick={() => { clearAllHistory(); setSuggestions([]); }}
                  className="flex items-center gap-1 text-[11px] text-black/25 hover:text-red-400 transition-colors"
                >
                  <Trash2 size={10} />
                  Очистить
                </button>
              )}
            </div>
            {suggestions.length === 0 && (
              <div className="py-6 text-center text-[12px] text-black/25">
                История пуста
              </div>
            )}
            {suggestions.map((s, idx) => {
              const Icon = s.type === 'history' || s.type === 'recent_entity' ? Clock : s.type === 'client' ? User : FileText;
              return (
                <div key={s.id} className="flex items-center group">
                  <button
                    data-idx={idx}
                    type="button"
                    onClick={() => handleSuggestionPick(s)}
                    className={[
                      'flex flex-1 min-w-0 items-center gap-3 rounded-[var(--app-radius-md)]',
                      'px-3 py-2.5 text-left text-[length:var(--text-sm)] transition-colors duration-150',
                      'hover:bg-[color-mix(in_srgb,var(--app-accent)_6%,var(--app-surface-soft))]',
                      activeIdx === idx
                        ? 'bg-[color-mix(in_srgb,var(--app-accent)_6%,var(--app-surface-soft))]'
                        : '',
                    ].join(' ')}
                  >
                    <Icon size={13} className="shrink-0 text-[color:var(--text-muted)] opacity-50" />
                    <span className="flex-1 truncate text-[color:var(--text-primary)]">{s.text}</span>
                    {s.subtitle && (
                      <span className="shrink-0 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
                        {s.subtitle}
                      </span>
                    )}
                  </button>
                  {(s.type === 'history') && (
                    <button
                      type="button"
                      onClick={() => { deleteHistoryItem(s.text); setSuggestions(prev => prev.filter(x => x.id !== s.id)); }}
                      className="shrink-0 p-1.5 mx-1 rounded-lg opacity-0 group-hover:opacity-100 text-black/20 hover:text-red-400 hover:bg-red-50 transition-all"
                      title="Удалить"
                    >
                      <X size={12} />
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Empty state hint */}
        {!query.trim() && suggestions.length === 0 && (
          <EmptyStatePanel>{t('globalSearch.hintEmpty')}</EmptyStatePanel>
        )}

        {/* AI answer placeholder (phase 2) — reserved space */}
        {query.trim() && (hasResults || aiLoading) && (
          <div
            className={[
              'min-h-[2.5rem] rounded-[var(--app-radius-md)] border border-[var(--app-border)]',
              'bg-[color-mix(in_srgb,var(--app-accent)_4%,var(--app-surface-soft))]',
              'px-4 py-3 text-[length:var(--text-sm)] transition-all duration-300',
              !aiAnswer && !aiLoading ? 'opacity-0 pointer-events-none' : 'opacity-100',
            ].join(' ')}
            aria-live="polite"
          >
            {aiLoading && (
              <span className="text-[color:var(--text-muted)] italic">
                {t('globalSearch.loading')}
              </span>
            )}
            {aiAnswer && (
              <span className="text-[color:var(--text-primary)]">{aiAnswer}</span>
            )}
          </div>
        )}

        {ftsLoading && <EmptyStatePanel>{t('globalSearch.loading')}</EmptyStatePanel>}

        {!ftsLoading && query.trim() && !hasResults && !suggestMode && (
          <EmptyStatePanel>{t('globalSearch.noResults')}</EmptyStatePanel>
        )}

        {/* Results */}
        <div ref={listRef} className="max-h-[60vh] space-y-4 overflow-y-auto pr-1 custom-scroll">
          {grouped.map(([section, items]) => (
            <GlobalSearchResultsSection
              key={section}
              title={t(sectionLabelKey(section))}
              count={items.length}
            >
              {items.map((item) => {
                const flatIdx = flatList.indexOf(item);
                return (
                  <div key={item.id} data-idx={flatIdx}>
                    <GlobalSearchResultRow
                      item={item}
                      active={activeIdx === flatIdx}
                      onPick={() => handlePick(item)}
                    />
                  </div>
                );
              })}
            </GlobalSearchResultsSection>
          ))}
        </div>
      </div>
    </Modal>
  );
}
