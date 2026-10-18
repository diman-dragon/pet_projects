// src/components/common/SearchHistoryDropdown.tsx
//
// Универсальный выпадающий список истории поиска/чата.
// Используется в FullTextSearchHeader, SemanticComposer, ChatInputBar, PopupInputBar.

import { Clock, X } from 'lucide-react';

export interface HistoryItem {
  key: string;
  text: string;
}

interface SearchHistoryDropdownProps {
  items: HistoryItem[];
  onSelect: (text: string) => void;
  onClear?: () => void;
  label?: string;
  className?: string;
  maxWidth?: string;
}

export function SearchHistoryDropdown({
  items,
  onSelect,
  onClear,
  label = 'История',
  className = '',
  maxWidth = 'w-full',
}: SearchHistoryDropdownProps) {
  if (items.length === 0) return null;

  return (
    <div
      className={[
        'absolute z-50 top-full left-0 mt-1 bg-white border border-black/[0.08]',
        'rounded-xl shadow-[0_8px_24px_rgba(0,0,0,0.10)] overflow-hidden',
        maxWidth,
        className,
      ].join(' ')}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-black/[0.05]">
        <span className="flex items-center gap-1.5 text-[11px] font-medium text-black/35 uppercase tracking-wide">
          <Clock size={11} />
          {label}
        </span>
        {onClear && (
          <button
            type="button"
            onClick={onClear}
            className="text-[11px] text-black/30 hover:text-black/50 transition-colors"
          >
            Очистить
          </button>
        )}
      </div>

      {/* Items */}
      <ul className="py-1">
        {items.map((item) => (
          <li key={item.key}>
            <button
              type="button"
              onClick={() => onSelect(item.text)}
              className="flex items-center gap-2.5 w-full px-3 py-2 text-left hover:bg-[var(--app-accent)]/5 transition-colors group"
            >
              <Clock size={13} className="shrink-0 text-black/20 group-hover:text-[var(--app-accent)]/50 transition-colors" />
              <span className="flex-1 truncate text-[length:var(--text-sm)] text-black/70 group-hover:text-black/90 transition-colors">
                {item.text}
              </span>
              <X
                size={11}
                className="shrink-0 opacity-0 group-hover:opacity-40 transition-opacity text-black/50"
              />
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
