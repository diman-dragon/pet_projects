// src/components/ui/form/Textarea.tsx
import { TextareaHTMLAttributes, forwardRef } from 'react';

type TextareaProps = TextareaHTMLAttributes<HTMLTextAreaElement> & {
  error?: boolean;
  rows?: number;
};

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ error, rows = 4, className = '', ...props }, ref) => (
    <div className="relative group w-full">
      <textarea
        ref={ref}
        rows={rows}
        {...props}
        className={[
          // База Win11: Mica-эффект и специфическая развесовка границ
          'w-full px-3 py-2 rounded-[6px] text-[length:var(--text-sm)] font-medium outline-none transition-all duration-200',
          'bg-[var(--app-surface)]/90 border border-black/[0.08] border-b-black/[0.4] text-black/85',
          'placeholder:text-black/30 placeholder:font-normal leading-[var(--leading-body)] resize-none',

          // Состояния
          error
            ? 'bg-red-50/30 border-red-400 focus:border-red-500'
            : 'hover:bg-white hover:border-black/[0.12] focus:bg-white',

          className,
        ]
          .filter(Boolean)
          .join(' ')}
      />

      {/* Акцентная полоска Win11 (Focus Indicator) */}
      <div
        className={[
          'absolute bottom-0 left-0 right-0 h-[2px] scale-x-0 transition-transform duration-300 origin-center rounded-b-[6px] pointer-events-none',
          error ? 'bg-red-600' : 'bg-[var(--app-accent)]',
          'group-focus-within:scale-x-100',
        ].join(' ')}
      />

      {/* Верхний блик (Elevation) */}
      <div className="absolute inset-x-0 top-0 h-[1px] bg-white/40 pointer-events-none rounded-t-[6px]" />
    </div>
  ),
);

Textarea.displayName = 'Textarea';
