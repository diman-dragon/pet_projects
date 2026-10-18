import { Brain, Trash2 } from 'lucide-react';
import type { TranslationKey } from '../../../../i18n/translations';

export function SemanticHeader({
  hasMessages,
  onClear,
  t,
}: {
  hasMessages: boolean;
  onClear: () => void;
  t: (key: TranslationKey) => string;
}) {
  return (
    <div className="flex items-center justify-between border-b border-black/[0.06] px-6 py-4">
      <div className="flex items-center gap-2">
        <Brain className="h-5 w-5 text-[var(--app-accent)]" />
        <h2 className="text-sm font-semibold text-black/80">{t('knowledge.semantic.title')}</h2>
      </div>
      {hasMessages ? (
        <button
          type="button"
          onClick={onClear}
          className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs text-black/55 transition-colors hover:bg-red-50 hover:text-red-500"
        >
          <Trash2 className="h-3.5 w-3.5" />
          {t('common.clear')}
        </button>
      ) : null}
    </div>
  );
}
