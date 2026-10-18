import { MarkdownMessage } from '../../../laila/components/shared/MarkdownMessage';
import type { ChatMessage } from '../../hooks/useKnowledgeHub';
import type { KnowledgeSourceSelection } from '../../types/source';
import { toKnowledgeSourceSelection } from './semanticSourceSelection';

export function extractCitationIndices(content: string): Set<number> {
  const indices = new Set<number>();
  const regex = /\[(\d+)\.?\]/g;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(content)) !== null) {
    indices.add(Number.parseInt(match[1], 10));
  }

  return indices;
}

export function MarkdownMessageWithCitations({
  content,
  sources,
  onSelectSource,
}: {
  content: string;
  sources?: ChatMessage['sources'];
  onSelectSource: (source: KnowledgeSourceSelection) => void;
}) {
  if (!content) {
    return null;
  }

  if (!sources?.length) {
    return <MarkdownMessage content={content} />;
  }

  const nodes: React.ReactNode[] = [];
  const citationRegex = /\[(\d+)\.?\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = citationRegex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(
        <MarkdownMessage key={`md-${lastIndex}`} content={content.slice(lastIndex, match.index)} />,
      );
    }

    const sourceIndex = Number.parseInt(match[1], 10) - 1;
    const source = sources[sourceIndex];
    nodes.push(
      <button
        key={`cite-${match.index}`}
        type="button"
        disabled={!source}
        onClick={() => {
          if (!source) {
            return;
          }
          onSelectSource(toKnowledgeSourceSelection(source));
        }}
        className={`mx-0.5 inline-flex items-center rounded px-1.5 py-0.5 text-xs font-semibold ${
          source
            ? 'bg-[var(--app-accent)]/10 text-[var(--app-accent)] hover:bg-[var(--app-accent)]/20'
            : 'bg-black/[0.06] text-black/40'
        }`}
      >
        [{match[1]}]
      </button>,
    );

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < content.length) {
    nodes.push(<MarkdownMessage key={`md-tail-${lastIndex}`} content={content.slice(lastIndex)} />);
  }

  return <div className="space-y-0 [&_.prose-chunk]:inline">{nodes}</div>;
}
