// src/components/ui/form/SearchInput.tsx
import { InputHTMLAttributes, forwardRef } from 'react';
import { Search, X } from 'lucide-react';

type SearchInputProps = Omit<InputHTMLAttributes<HTMLInputElement>, 'onChange'> & {
  value: string;
  onChange: (value: string) => void;
  onClear?: () => void;
};

export const SearchInput = forwardRef<HTMLInputElement, SearchInputProps>(
  ({ value, onChange, onClear, className = '', placeholder = '', ...props }, ref) => {
    const handleClear = () => {
      onChange('');
      onClear?.();
    };

    return (
      <div className="relative group flex items-center w-full overflow-hidden">
        {/* Иконка поиска: Win11 Style — чуть тоньше и бледнее */}
        <div className="absolute left-2.5 flex items-center justify-center pointer-events-none z-10">
          <Search
            size={14}
            strokeWidth={2.2}
            className="text-[var(--app-text-muted)] group-focus-within:text-[var(--app-accent)] transition-colors duration-200"
          />
        </div>

        <input
          ref={ref}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          {...props}
          className={[
            // База: Мягкий фон (Mica/Acrylic effect), тонкая граница
            // Используем системные токены для шрифтов: размер, вес, межстрочное
            'w-full h-8 pl-8 pr-8 rounded-[6px]',
            'text-[length:var(--text-sm)] leading-[var(--leading-body)] font-medium',
            'bg-[color-mix(in_srgb,var(--app-text)_4%,transparent)] border border-[var(--app-border)] text-[var(--app-text)]',
            // Плейсхолдер: тот же шрифт что и текст, но с прозрачностью как у secondary text
            'placeholder:text-[var(--app-text-secondary)] placeholder:font-medium placeholder:opacity-70',

            'hover:bg-[color-mix(in_srgb,var(--app-text)_6%,transparent)] hover:border-[var(--app-border-strong)]',
            'focus:bg-[var(--app-surface-raised)] focus:border-[var(--app-border)] focus:shadow-[var(--app-shadow-sm)]',
            className,
          ]
            .filter(Boolean)
            .join(' ')}
        />

        {/* Кнопка очистки: Появляется только при наличии текста */}
        <div className="absolute right-1 flex items-center">
          {value && (
            <button
              type="button"
              onClick={handleClear}
              className="p-1.5 mr-1 rounded-md text-[var(--app-text-muted)] hover:text-[var(--app-text)] hover:bg-[color-mix(in_srgb,var(--app-text)_6%,transparent)] active:scale-90 transition-all duration-150"
            >
              <X size={14} strokeWidth={2.5} />
            </button>
          )}
        </div>

        {/* Акцентная полоска (Win11 Accent Line) */}
        <div
          className={[
            'absolute bottom-0 left-0 right-0 h-[2px] bg-[var(--app-accent)]',
            'scale-x-0 group-focus-within:scale-x-100 transition-transform duration-300 ease-out',
            'origin-center rounded-b-[6px]',
          ].join(' ')}
        />

        <div className="pointer-events-none absolute inset-x-0 top-0 h-px rounded-t-[6px] bg-[color-mix(in_srgb,var(--app-surface-raised)_55%,transparent)]" />
      </div>
    );
  },
);

SearchInput.displayName = 'SearchInput';
