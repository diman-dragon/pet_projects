import { useState } from 'react';
import { Sparkles, RefreshCw, Loader2 } from 'lucide-react';
import Markdown from 'react-markdown';
import { useI18n } from '@/i18n/AppLanguageProvider';
import { UiText } from '@/components/ui';

interface AISummaryBlockProps {
  summary: string | null | undefined;
  label?: string;
  onGenerate: () => Promise<void>;
  streaming?: string;
  className?: string;
  loading?: boolean;
}

export function AISummaryBlock({
  summary,
  label,
  onGenerate,
  streaming,
  className = '',
  loading: externalLoading,
}: AISummaryBlockProps) {
  const { t } = useI18n();
  const resolvedLabel = label ?? t('aiSummary.title');
  const [internalLoading, setInternalLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loading = externalLoading ?? internalLoading;

  const handleGenerate = async () => {
    setInternalLoading(true);
    setError(null);

    try {
      await onGenerate();
    } catch (e: unknown) {
      const err = e as { name?: string; message?: string };
      if (err?.name !== 'AbortError') {
        setError(err?.message ?? t('aiSummary.errorGeneric'));
      }
    } finally {
      setInternalLoading(false);
    }
  };

  const displayText = loading && streaming ? streaming : summary;
  const hasSummary = Boolean(displayText);

  return (
    <div
      className={`space-y-3 rounded-xl border p-4 ${
        hasSummary
          ? 'border-[var(--app-border)] bg-[color-mix(in_srgb,var(--app-text)_2%,transparent)]'
          : 'border-dashed border-[var(--app-border)] bg-transparent'
      } ${className}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-[var(--app-text)]">
            <Sparkles size={11} className="text-[var(--app-surface-raised)]" />
          </div>
          <UiText
            as="span"
            size="xs"
            weight="semibold"
            tone="muted"
            className="leading-[var(--leading-heading)]"
          >
            {resolvedLabel}
          </UiText>
        </div>

        <button
          type="button"
          onClick={handleGenerate}
          disabled={loading}
          className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-[length:var(--text-xs)] font-medium leading-[var(--leading-body)] text-[var(--app-text-secondary)] transition-all hover:bg-[color-mix(in_srgb,var(--app-text)_6%,transparent)] hover:text-[var(--app-text)] disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
          {hasSummary ? t('aiSummary.refresh') : t('aiSummary.generate')}
        </button>
      </div>

      {loading && !streaming && (
        <div className="flex items-center gap-2 text-[var(--app-text-muted)]">
          <Loader2 size={13} className="animate-spin" />
          <UiText as="span" size="xs" tone="muted">
            {t('aiSummary.generating')}
          </UiText>
        </div>
      )}

      {displayText && (
        <div className="prose-custom max-w-none text-[var(--app-text-secondary)]">
          <Markdown
            components={{
              h1: ({node, ...props}) => <UiText size="md" weight="black" className="mt-4 mb-2 block" {...props} />,
              h2: ({node, ...props}) => <UiText size="sm" weight="bold" className="mt-3 mb-1 block" {...props} />,
              p: ({node, ...props}) => <UiText size="xs" className="mb-2 leading-relaxed block" {...props} />,
              ul: ({node, ...props}) => <ul className="list-disc pl-4 mb-3 space-y-1 text-[var(--text-xs)]" {...props} />,
              ol: ({node, ...props}) => <ol className="list-decimal pl-4 mb-3 space-y-1 text-[var(--text-xs)]" {...props} />,
              li: ({node, ...props}) => <li className="leading-snug" {...props} />,
              strong: ({node, ...props}) => <UiText as="strong" weight="bold" className="text-[var(--app-text)]" {...props} />,
            }}
          >
            {displayText}
          </Markdown>
          {loading && (
            <span className="ml-1 inline-block h-3.5 w-0.5 animate-pulse bg-[var(--app-accent)] align-middle" />
          )}
        </div>
      )}

      {!hasSummary && !loading && (
        <UiText as="p" size="xs" tone="muted" className="italic">
          {t('aiSummary.empty')}
        </UiText>
      )}

      {error && (
        <UiText
          as="p"
          size="xs"
          className="text-[color-mix(in_srgb,var(--app-danger)_80%,var(--app-text))]"
        >
          {t('aiSummary.errorPrefix')} {error}
        </UiText>
      )}
    </div>
  );
}
