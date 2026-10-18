import { forwardRef, type InputHTMLAttributes, type ReactNode } from 'react';

export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  size?: 'sm' | 'md';
  error?: string | boolean;
  icon?: ReactNode;
  label?: string;
  hint?: string;
}

const sizeClass: Record<NonNullable<InputProps['size']>, string> = {
  sm: 'h-8 rounded-[var(--app-radius-sm)] px-3 text-[length:var(--text-sm)]',
  md: 'h-9 rounded-[var(--app-radius-md)] px-3.5 text-[length:var(--text-sm)]',
};

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ size = 'md', error, icon, label, hint, className = '', ...props }, ref) => (
    <div className="flex w-full flex-col gap-1.5">
      {label ? (
        <label className="label-caps">
          {label}
        </label>
      ) : null}
      <div className="group relative">
        {icon ? (
          <span className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-[var(--app-text-muted)]">
            {icon}
          </span>
        ) : null}
        <input
          ref={ref}
          className={[
            'w-full border bg-[var(--app-surface)] font-medium leading-[var(--leading-body)] text-[var(--app-text)] outline-none transition-all duration-150',
            'placeholder:text-[var(--app-text-muted)]',
            'focus:border-[var(--app-accent)] focus:ring-2 focus:ring-[color-mix(in_srgb,var(--app-accent)_18%,transparent)]',
            'disabled:cursor-not-allowed disabled:bg-[var(--app-surface-soft)] disabled:text-[var(--app-text-muted)]',
            error
              ? 'border-[var(--app-danger)] focus:border-[var(--app-danger)] focus:ring-[color-mix(in_srgb,var(--app-danger)_16%,transparent)]'
              : 'border-[var(--app-border)] hover:border-[var(--app-border-strong)]',
            icon ? 'pl-9' : '',
            sizeClass[size],
            className,
          ].join(' ')}
          {...props}
        />
      </div>
      {typeof error === 'string' ? (
        <p className="text-[length:var(--text-xs)] leading-[var(--leading-body)] text-[var(--app-danger)]">
          {error}
        </p>
      ) : hint ? (
        <p className="text-[length:var(--text-xs)] leading-[var(--leading-body)] text-[var(--app-text-muted)]">
          {hint}
        </p>
      ) : null}
    </div>
  ),
);

Input.displayName = 'Input';
