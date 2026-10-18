import type { ChatMessage } from '../../hooks/useKnowledgeHub';
import type { KnowledgeSourceSelection } from '../../types/source';
import { toKnowledgeSourceSelection } from './semanticSourceSelection';

export function SemanticMessageSources({
  sources,
  selectedSource,
  onSelectSource,
  title,
}: {
  sources: NonNullable<ChatMessage['sources']>;
  selectedSource: KnowledgeSourceSelection | null;
  onSelectSource: (source: KnowledgeSourceSelection) => void;
  title: string;
}) {
  if (sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-4 border-t border-black/[0.06] pt-3">
      <p className="mb-2 text-[10px] uppercase tracking-wider text-black/40">{title}</p>
      <div className="flex flex-wrap gap-1.5">
        {sources.map((source) => (
          <button
            key={source.index}
            type="button"
            onClick={() => onSelectSource(toKnowledgeSourceSelection(source))}
            className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs transition-colors ${
              selectedSource?.index === source.index
                ? 'bg-[var(--app-accent)]/20 text-[var(--app-accent)]'
                : 'bg-black/[0.03] text-black/60 hover:bg-black/[0.06]'
            }`}
          >
            <span className="font-medium">[{source.index}]</span>
            <span className="max-w-[150px] truncate">{source.title}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
