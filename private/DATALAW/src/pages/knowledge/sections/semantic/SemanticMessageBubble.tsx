import { useMemo } from 'react';
import type { ChatMessage } from '../../hooks/useKnowledgeHub';
import type { KnowledgeSourceSelection } from '../../types/source';
import { extractCitationIndices, MarkdownMessageWithCitations } from './semanticCitations';
import { SemanticMessageSources } from './SemanticMessageSources';
import { SemanticThinkingAnimation } from './SemanticThinkingAnimation';

interface SemanticMessageBubbleProps {
  message: ChatMessage;
  isLastMessage: boolean;
  isStreaming: boolean;
  selectedSource: KnowledgeSourceSelection | null;
  onSelectSource: (source: KnowledgeSourceSelection) => void;
  sourcesTitle: string;
  thinkingLabel?: string;
}

export function SemanticMessageBubble({
  message,
  isLastMessage,
  isStreaming,
  selectedSource,
  onSelectSource,
  sourcesTitle,
  thinkingLabel = 'Thinking...',
}: SemanticMessageBubbleProps) {
  const isUser = message.role === 'user';
  const usedCitationIndices = useMemo(
    () =>
      message.role === 'assistant' ? extractCitationIndices(message.content) : new Set<number>(),
    [message.content, message.role],
  );

  if (!message.content && message.role === 'assistant' && isLastMessage && isStreaming) {
    return (
      <div className="flex justify-start">
        <div className="max-w-[85%] rounded-2xl border border-black/[0.06] bg-white">
          <SemanticThinkingAnimation label={thinkingLabel} />
        </div>
      </div>
    );
  }

  if (!message.content && message.role === 'assistant') {
    return null;
  }

  const visibleSources =
    usedCitationIndices.size > 0
      ? (message.sources?.filter((source) => usedCitationIndices.has(source.index)) ?? [])
      : (message.sources ?? []);

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-5 py-4 ${
          isUser ? 'bg-[var(--app-accent)] text-white' : 'border border-black/[0.06] bg-white'
        }`}
      >
        <div className={`text-sm leading-relaxed ${isUser ? 'text-white' : 'text-black/80'}`}>
          {message.role === 'assistant' ? (
            <MarkdownMessageWithCitations
              content={message.content}
              sources={message.sources}
              onSelectSource={onSelectSource}
            />
          ) : (
            message.content
          )}
        </div>

        {message.role === 'assistant' && visibleSources.length > 0 ? (
          <SemanticMessageSources
            sources={visibleSources}
            selectedSource={selectedSource}
            onSelectSource={onSelectSource}
            title={sourcesTitle}
          />
        ) : null}
      </div>
    </div>
  );
}
