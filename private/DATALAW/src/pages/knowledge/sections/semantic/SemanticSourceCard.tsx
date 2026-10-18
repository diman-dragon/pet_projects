import { FileText } from 'lucide-react';
import { openPath } from '@tauri-apps/plugin-opener';
import type { TranslationKey } from '../../../../i18n/translations';
import type { KnowledgeSourceSelection } from '../../types/source';

function HighlightedExcerpt({ text }: { text: string }) {
  return (
    <p className="whitespace-pre-wrap rounded-r-lg border-l-2 border-[var(--app-accent)] bg-[var(--app-accent)]/5 py-1 pl-3 text-sm leading-relaxed text-black/80">
      {text}
    </p>
  );
}

export function SemanticSourceCard({
  selectedSource,
  t,
}: {
  selectedSource: KnowledgeSourceSelection;
  t: (key: TranslationKey, params?: Record<string, string | number>) => string;
}) {
  const fileName = selectedSource.filePath.split(/[/\\]/).pop() || selectedSource.filePath;

  return (
    <div className="space-y-3">
      <div className="flex items-start gap-2">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded bg-[var(--app-accent)]/10 text-xs font-semibold text-[var(--app-accent)]">
          {selectedSource.index}
        </span>
        <div className="min-w-0 flex-1">
          {selectedSource.articleRef ? (
            <span className="mb-2 inline-flex items-center rounded-md bg-[var(--app-accent)]/10 px-2 py-0.5 text-xs font-mono font-semibold text-[var(--app-accent)]">
              {selectedSource.articleRef}
            </span>
          ) : null}
          <h4 className="text-sm font-medium text-black/80">{selectedSource.title}</h4>
          <p className="mt-0.5 truncate text-xs text-black/40" title={selectedSource.filePath}>
            {fileName}
          </p>
        </div>
      </div>

      <div className="rounded-2xl border border-black/[0.06] bg-white p-4 shadow-[0_1px_2px_rgba(0,0,0,0.03)]">
        <HighlightedExcerpt text={selectedSource.excerpt} />
      </div>

      <button
        type="button"
        onClick={() => openPath(selectedSource.filePath)}
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-[var(--app-accent)]/10 px-3 py-2 text-sm font-medium text-[var(--app-accent)] transition-colors hover:bg-[var(--app-accent)]/15"
      >
        <FileText className="h-4 w-4" />
        {t('knowledge.semantic.openFile')}
      </button>
    </div>
  );
}
