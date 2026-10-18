// src/components/layout/Header/global-search/GlobalSearchResultRow.tsx
//
// ИТЕРАЦИЯ 4: добавлено prop active (keyboard focus), breadcrumbs.

import { BookOpen, Building2, FileText, Mail, Search, Users } from 'lucide-react';
import { UiText } from '../../../ui/atoms/UiText';
import type { GlobalSearchResult } from '../../../../services/search/GlobalSearchService';

export type GlobalSearchResultRowProps = {
  item: GlobalSearchResult;
  onPick: () => void;
  active?: boolean;
};

function SectionIcon({ section }: { section: GlobalSearchResult['section'] }) {
  const cls = 'text-[var(--app-accent)]';
  const size = 14;
  switch (section) {
    case 'clients':   return <Building2 size={size} strokeWidth={2} className={cls} />;
    case 'cases':     return <FileText  size={size} strokeWidth={2} className={cls} />;
    case 'appeals':   return <Mail      size={size} strokeWidth={2} className={cls} />;
    case 'knowledge': return <BookOpen  size={size} strokeWidth={2} className={cls} />;
    case 'users':     return <Users     size={size} strokeWidth={2} className={cls} />;
    default:          return <Search    size={size} strokeWidth={2} className={cls} />;
  }
}

export function GlobalSearchResultRow({ item, onPick, active = false }: GlobalSearchResultRowProps) {
  return (
    <button
      type="button"
      onClick={onPick}
      className={[
        'flex w-full items-start gap-3 rounded-[var(--app-radius-md)] border border-[var(--app-border)]',
        'bg-[var(--app-surface-raised)] px-4 py-3 text-left transition-colors duration-150',
        'hover:border-[color-mix(in_srgb,var(--app-accent)_35%,var(--app-border))]',
        'hover:bg-[color-mix(in_srgb,var(--app-accent)_6%,var(--app-surface-soft))]',
        active
          ? 'border-[color-mix(in_srgb,var(--app-accent)_35%,var(--app-border))] bg-[color-mix(in_srgb,var(--app-accent)_6%,var(--app-surface-soft))]'
          : '',
      ].join(' ')}
    >
      <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-[var(--app-radius-sm)] bg-[color-mix(in_srgb,var(--app-accent)_12%,transparent)]">
        <SectionIcon section={item.section} />
      </div>
      <div className="min-w-0 flex-1 space-y-0.5">
        <UiText as="div" size="sm" weight="semibold" className="truncate">
          {item.title}
        </UiText>
        {/* Breadcrumb (Категория → Папка → Файл) если есть, иначе subtitle */}
        {item.breadcrumb ? (
          <UiText as="div" size="xs" tone="muted" className="truncate">
            {item.breadcrumb}
          </UiText>
        ) : (
          <UiText as="div" size="xs" tone="secondary" className="line-clamp-1">
            {item.subtitle}
          </UiText>
        )}
      </div>
      {item.source === 'semantic' && (
        <span className="mt-1 shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium bg-[color-mix(in_srgb,var(--app-accent)_15%,transparent)] text-[var(--app-accent)]">
          AI
        </span>
      )}
    </button>
  );
}
