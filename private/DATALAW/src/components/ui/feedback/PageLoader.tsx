// src/components/ui/feedback/PageLoader.tsx
import { Loader2 } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';

interface PageLoaderProps {
  text?: string;
  fullHeight?: boolean;
  className?: string;
}

export function PageLoader({ text, fullHeight = false, className = '' }: PageLoaderProps) {
  const { t } = useI18n();
  const label = text ?? t('loader.loading');

  return (
    <div
      className={[
        'flex flex-col items-center justify-center gap-3 transition-all duration-500',
        fullHeight ? 'h-screen w-full' : 'h-64 w-full',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <div className="relative flex items-center justify-center">
        <div className="absolute inset-0 animate-pulse rounded-full bg-blue-500/5 blur-xl" />

        <Loader2 size={24} strokeWidth={2.2} className="animate-spin text-blue-600 opacity-80" />
      </div>

      <span className="select-none animate-pulse text-[length:var(--text-sm)] font-medium leading-[var(--leading-body)] tracking-tight text-[var(--app-text-muted)]">
        {label}
      </span>
    </div>
  );
}
