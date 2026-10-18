import { Sparkles } from 'lucide-react';
import type { TranslationKey } from '../../../../i18n/translations';

interface SemanticEmptyStateProps {
  onPickSuggestion: (value: string) => void;
  t: (key: TranslationKey) => string;
}

const SUGGESTION_KEYS = [
  'knowledge.semantic.suggestion.1',
  'knowledge.semantic.suggestion.2',
  'knowledge.semantic.suggestion.3',
] satisfies TranslationKey[];

export function SemanticEmptyState({ onPickSuggestion, t }: SemanticEmptyStateProps) {
  return (
    <div className="flex h-full flex-col items-center justify-center space-y-4 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[var(--app-accent)]/20 to-[var(--app-accent)]/5">
        <Sparkles className="h-8 w-8 text-[var(--app-accent)]" />
      </div>
      <div className="max-w-md space-y-2">
        <h3 className="text-lg font-semibold text-black/80">
          {t('knowledge.semantic.emptyTitle')}
        </h3>
        <p className="text-sm text-black/50">{t('knowledge.semantic.emptyHint')}</p>
      </div>
      <div className="flex max-w-lg flex-wrap justify-center gap-2">
        {SUGGESTION_KEYS.map((key) => {
          const value = t(key);
          return (
            <button
              key={key}
              type="button"
              onClick={() => onPickSuggestion(value)}
              className="rounded-full bg-black/[0.03] px-4 py-2 text-xs text-black/60 transition-colors hover:bg-black/[0.06]"
            >
              {value}
            </button>
          );
        })}
      </div>
    </div>
  );
}
