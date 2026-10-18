import { BarChart2, BookOpen, Clock, FileText, Globe, Hash, RotateCcw, Tag } from 'lucide-react';
import type { TranslationKey } from '../../../../i18n/translations';
import { FilterRow } from './FilterRow';
import { FILE_TYPES, TOP_K_OPTIONS } from './constants';
import type { FilterState } from './types';

const LANGUAGE_OPTIONS = [
  { value: '', key: 'knowledge.fulltext.languages.all' },
  { value: 'ru', key: 'language.ru' },
  { value: 'en', key: 'language.en' },
  { value: 'sr', key: 'language.sr' },
] satisfies Array<{ value: string; key: TranslationKey }>;

const MATCH_TYPE_OPTIONS = [
  { id: 'all', key: 'knowledge.fulltext.matchType.all' },
  { id: 'text', key: 'knowledge.fulltext.matchType.text' },
  { id: 'vector', key: 'knowledge.fulltext.matchType.vector' },
  { id: 'hybrid', key: 'knowledge.fulltext.matchType.hybrid' },
] satisfies Array<{ id: FilterState['matchType']; key: TranslationKey }>;

interface FullTextFiltersPanelProps {
  filters: FilterState;
  folders: string[];
  categories: string[];
  languages: string[];
  hasActiveFilters: boolean;
  onChange: (patch: Partial<FilterState>) => void;
  onReset: () => void;
  t: (key: TranslationKey, params?: Record<string, string | number>) => string;
}

export function FullTextFiltersPanel({
  filters,
  folders,
  categories,
  languages,
  hasActiveFilters,
  onChange,
  onReset,
  t,
}: FullTextFiltersPanelProps) {
  return (
    <div className="border-t border-black/[0.05] bg-black/[0.015] px-5 py-1">
      <FilterRow
        icon={<BookOpen className="h-3.5 w-3.5" />}
        label={t('knowledge.fulltext.filters.titleLabel')}
      >
        <input
          type="text"
          value={filters.titleQuery}
          onChange={(event) => onChange({ titleQuery: event.target.value })}
          placeholder={t('knowledge.fulltext.filters.titlePlaceholder')}
          className="w-52 rounded-lg border border-black/[0.08] bg-white px-2.5 py-1.5 text-xs outline-none focus:border-[var(--app-accent)]/40"
        />
      </FilterRow>

      <FilterRow
        icon={<Tag className="h-3.5 w-3.5" />}
        label={t('knowledge.fulltext.filters.categoryLabel')}
      >
        <select
          value={filters.category}
          onChange={(event) => onChange({ category: event.target.value })}
          className="w-52 rounded-lg border border-black/[0.08] bg-white px-2.5 py-1.5 text-xs outline-none"
        >
          <option value="">{t('knowledge.fulltext.categories.all')}</option>
          {categories.map((category) => (
            <option key={category} value={category}>
              {category}
            </option>
          ))}
        </select>
      </FilterRow>

      <FilterRow
        icon={<Globe className="h-3.5 w-3.5" />}
        label={t('knowledge.fulltext.filters.languageLabel')}
      >
        <select
          value={filters.language}
          onChange={(event) => onChange({ language: event.target.value })}
          className="w-52 rounded-lg border border-black/[0.08] bg-white px-2.5 py-1.5 text-xs outline-none"
        >
          {LANGUAGE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {t(option.key)}
            </option>
          ))}
          {languages
            .filter((language) => !LANGUAGE_OPTIONS.some((option) => option.value === language))
            .map((language) => (
              <option key={language} value={language}>
                {language.toUpperCase()}
              </option>
            ))}
        </select>
      </FilterRow>

      <FilterRow
        icon={<BookOpen className="h-3.5 w-3.5" />}
        label={t('knowledge.fulltext.filters.folderLabel')}
      >
        <select
          value={filters.folder}
          onChange={(event) => onChange({ folder: event.target.value })}
          className="w-52 rounded-lg border border-black/[0.08] bg-white px-2.5 py-1.5 text-xs outline-none"
        >
          <option value="">{t('knowledge.fulltext.folders.all')}</option>
          {folders.map((folder) => (
            <option key={folder} value={folder}>
              {folder}
            </option>
          ))}
        </select>
      </FilterRow>

      <FilterRow
        icon={<FileText className="h-3.5 w-3.5" />}
        label={t('knowledge.fulltext.filters.fileTypeLabel')}
      >
        <div className="flex flex-wrap gap-1">
          <ToggleChip active={!filters.fileType} onClick={() => onChange({ fileType: '' })}>
            {t('knowledge.fulltext.fileTypes.all')}
          </ToggleChip>
          {FILE_TYPES.map((ext) => (
            <ToggleChip
              key={ext}
              active={filters.fileType === ext}
              onClick={() => onChange({ fileType: filters.fileType === ext ? '' : ext })}
            >
              .{ext}
            </ToggleChip>
          ))}
        </div>
      </FilterRow>

      <FilterRow
        icon={<Hash className="h-3.5 w-3.5" />}
        label={t('knowledge.fulltext.filters.matchLabel')}
      >
        <div className="flex flex-wrap gap-1">
          {MATCH_TYPE_OPTIONS.map((option) => (
            <ToggleChip
              key={option.id}
              active={filters.matchType === option.id}
              onClick={() => onChange({ matchType: option.id })}
            >
              {t(option.key)}
            </ToggleChip>
          ))}
        </div>
      </FilterRow>

      <FilterRow
        icon={<BarChart2 className="h-3.5 w-3.5" />}
        label={t('knowledge.fulltext.filters.scoreLabel', { value: filters.minScore })}
      >
        <input
          type="range"
          min={0}
          max={90}
          step={5}
          value={filters.minScore}
          onChange={(event) => onChange({ minScore: Number(event.target.value) })}
          className="w-32 cursor-pointer accent-[var(--app-accent)]"
        />
      </FilterRow>

      <FilterRow
        icon={<Clock className="h-3.5 w-3.5" />}
        label={t('knowledge.fulltext.filters.topKLabel')}
      >
        <div className="flex items-center gap-3">
          <div className="flex gap-1">
            {TOP_K_OPTIONS.map((topK) => (
              <ToggleChip
                key={topK}
                active={filters.topK === topK}
                onClick={() => onChange({ topK })}
              >
                {topK}
              </ToggleChip>
            ))}
          </div>
          {hasActiveFilters ? (
            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center gap-1 rounded-md border border-dashed border-black/[0.12] px-2.5 py-1 text-[10px] text-black/45 transition-colors hover:border-black/[0.2] hover:text-black/70"
            >
              <RotateCcw className="h-3 w-3" />
              {t('common.clear')}
            </button>
          ) : null}
        </div>
      </FilterRow>
    </div>
  );
}

function ToggleChip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-md border px-2 py-1 text-[10px] transition-colors ${
        active
          ? 'border-transparent bg-[var(--app-accent)] text-white'
          : 'border-black/[0.08] bg-white text-black/55 hover:bg-black/[0.03]'
      }`}
    >
      {children}
    </button>
  );
}
