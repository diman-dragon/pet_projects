import { Search, X, ArrowRight, Loader2 } from 'lucide-react';

interface SearchInputProps {
  value: string;
  onChange: (v: string) => void;
  onClear: () => void;
  onSearch: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  isRunning: boolean;
}

const MIN_WORDS = 2;

export function SearchInput({ value, onChange, onClear, onSearch, onKeyDown, isRunning }: SearchInputProps) {
  const words = value.trim().split(/\s+/).filter(Boolean);
  const wordCount = words.length;
  const ready = wordCount >= MIN_WORDS;
  const typing = value.trim().length > 0 && !ready;

  return (
    <div className="p-3 space-y-2">
      <div className="relative flex items-center gap-2">
        {/* Input */}
        <div className="relative flex-1">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-black/25 pointer-events-none" />
          <input
            type="text"
            value={value}
            onChange={e => onChange(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Введите запрос и нажмите →"
            disabled={isRunning}
            className={`
              h-10 w-full rounded-xl border pl-9 pr-8
              text-[length:var(--text-xs)] text-black/80 bg-white
              transition-all duration-200 placeholder:text-black/20
              focus:outline-none focus:ring-2 disabled:opacity-50
              ${ready
                ? 'border-[var(--app-accent)]/40 focus:ring-[var(--app-accent)]/10'
                : 'border-black/[0.08] focus:ring-black/5'
              }
            `}
          />
          {value && !isRunning && (
            <button
              onClick={onClear}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-black/20 hover:text-black/50 transition-colors"
            >
              <X size={12} />
            </button>
          )}
        </div>

        {/* Кнопка поиска */}
        <button
          onClick={onSearch}
          disabled={!ready || isRunning}
          className={`
            flex h-10 w-10 shrink-0 items-center justify-center rounded-xl
            transition-all duration-200 font-medium
            ${ready && !isRunning
              ? 'bg-[var(--app-accent)] text-white shadow-sm hover:opacity-90 cursor-pointer'
              : 'bg-black/[0.06] text-black/25 cursor-not-allowed'
            }
          `}
          title="Начать поиск (Enter)"
        >
          {isRunning
            ? <Loader2 size={14} className="animate-spin" />
            : <ArrowRight size={14} />
          }
        </button>
      </div>

      {/* Подсказка */}
      {typing && (
        <div className="flex items-center gap-2 px-1">
          <div className="flex gap-1">
            {Array.from({ length: MIN_WORDS }).map((_, i) => (
              <div
                key={i}
                className={`h-0.5 w-6 rounded-full transition-all duration-300 ${
                  i < wordCount ? 'bg-[var(--app-accent)]/50' : 'bg-black/[0.08]'
                }`}
              />
            ))}
          </div>
          <span className="text-[9px] text-black/30">
            ещё {MIN_WORDS - wordCount} слов{MIN_WORDS - wordCount === 1 ? 'о' : 'а'} — потом Enter
          </span>
        </div>
      )}

      {ready && !isRunning && (
        <p className="text-[9px] text-black/25 px-1">
          Нажмите Enter или → для семантического поиска
        </p>
      )}
    </div>
  );
}
