import type { RefObject } from 'react';
import type { ChatMessage } from '../../hooks/useKnowledgeHub';
import type { TranslationKey } from '../../../../i18n/translations';
import type { KnowledgeSourceSelection } from '../../types/source';
import { SemanticEmptyState } from './SemanticEmptyState';
import { SemanticMessageBubble } from './SemanticMessageBubble';

export function SemanticMessagesPane({
  messages,
  isStreaming,
  selectedSource,
  onSelectSource,
  messagesEndRef,
  onPickSuggestion,
  t,
}: {
  messages: ChatMessage[];
  isStreaming: boolean;
  selectedSource: KnowledgeSourceSelection | null;
  onSelectSource: (source: KnowledgeSourceSelection) => void;
  messagesEndRef: RefObject<HTMLDivElement | null>;
  onPickSuggestion: (query: string) => void;
  t: (key: TranslationKey) => string;
}) {
  const hasMessages = messages.length > 0;

  return (
    <div className="flex-1 space-y-6 overflow-y-auto px-4 py-6">
      {hasMessages ? (
        messages.map((message, index) => (
          <SemanticMessageBubble
            key={message.id}
            message={message}
            isLastMessage={index === messages.length - 1}
            isStreaming={isStreaming}
            selectedSource={selectedSource}
            onSelectSource={onSelectSource}
            sourcesTitle={t('knowledge.semantic.sourcesUsed')}
            thinkingLabel={t('laila.thinking')}
          />
        ))
      ) : (
        <SemanticEmptyState onPickSuggestion={onPickSuggestion} t={t} />
      )}
      <div ref={messagesEndRef} />
    </div>
  );
}
