import type { SelectHTMLAttributes } from 'react';
import { ChevronDown } from 'lucide-react';

export interface SelectOption {
  label: string;
  value: string;
}

type SelectProps = Omit<SelectHTMLAttributes<HTMLSelectElement>, 'onChange' | 'value'> & {
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
  error?: string;
};

export function Select({
  value,
  onChange,
  options,
  error,
  className = '',
  disabled,
  ...props
}: SelectProps) {
  return (
    <div className="relative w-full">
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className={[
          'h-9 w-full appearance-none rounded-[var(--app-radius-md)] border bg-[var(--app-surface)] pl-3.5 pr-9',
          'text-[length:var(--text-sm)] font-medium leading-[var(--leading-body)] text-[var(--app-text)] outline-none transition-all duration-150',
          'disabled:cursor-not-allowed disabled:bg-[var(--app-surface-soft)] disabled:text-[var(--app-text-muted)]',
          error
            ? 'border-[var(--app-danger)] focus:border-[var(--app-danger)] focus:ring-2 focus:ring-[color-mix(in_srgb,var(--app-danger)_16%,transparent)]'
            : 'border-[var(--app-border)] hover:border-[var(--app-border-strong)] focus:border-[var(--app-accent)] focus:ring-2 focus:ring-[color-mix(in_srgb,var(--app-accent)_18%,transparent)]',
          className,
        ].join(' ')}
        {...props}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-[var(--app-text-muted)]">
        <ChevronDown size={16} strokeWidth={2} />
      </span>
    </div>
  );
}
