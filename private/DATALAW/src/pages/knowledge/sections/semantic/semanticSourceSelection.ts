import type { ChatMessage } from '../../hooks/useKnowledgeHub';
import type { KnowledgeSourceSelection } from '../../types/source';

type SourceLike = NonNullable<ChatMessage['sources']>[number];

export function toKnowledgeSourceSelection(source: SourceLike): KnowledgeSourceSelection {
  return {
    index: source.index,
    title: source.title,
    excerpt: source.excerpt,
    filePath: source.filePath,
    charStart: source.charStart,
    charEnd: source.charEnd,
    articleRef: source.articleRef,
  };
}
