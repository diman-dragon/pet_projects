import { useEffect, useRef, useState } from 'react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import type { ChatMessage } from '../../hooks/useKnowledgeHub';
import type { KnowledgeSourceSelection } from '../../types/source';
import { SemanticComposer } from './SemanticComposer';
import { SemanticHeader } from './SemanticHeader';
import { SemanticMessagesPane } from './SemanticMessagesPane';
import { SemanticSourceSidebar } from './SemanticSourceSidebar';

interface SemanticSearchTabProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  isThinking: boolean;
  onSendMessage: (query: string) => void;
  onAbort: () => void;
  onClear: () => void;
  onSelectSource: (source: KnowledgeSourceSelection) => void;
  selectedSource: KnowledgeSourceSelection | null;
}

export function SemanticSearchTab({
  messages,
  isStreaming,
  isThinking,
  onSendMessage,
  onAbort,
  onClear,
  onSelectSource,
  selectedSource,
}: SemanticSearchTabProps) {
  const { t } = useI18n();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(event.target.value);
    event.target.style.height = 'auto';
    event.target.style.height = `${Math.min(event.target.scrollHeight, 200)}px`;
  };

  const handleSubmit = () => {
    if (!input.trim() || isStreaming || isThinking) {
      return;
    }

    onSendMessage(input.trim());
    setInput('');

    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  };

  const hasMessages = messages.length > 0;

  return (
    <div className="flex h-full overflow-hidden">
      <div className="flex min-w-0 flex-1 flex-col">
        <SemanticHeader hasMessages={hasMessages} onClear={onClear} t={t} />

        <SemanticMessagesPane
          messages={messages}
          isStreaming={isStreaming}
          selectedSource={selectedSource}
          onSelectSource={onSelectSource}
          messagesEndRef={messagesEndRef}
          onPickSuggestion={onSendMessage}
          t={t}
        />

        <SemanticComposer
          input={input}
          isStreaming={isStreaming}
          isThinking={isThinking}
          inputRef={inputRef}
          onInputChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onSubmit={handleSubmit}
          onAbort={onAbort}
          t={t}
        />
      </div>

      <SemanticSourceSidebar selectedSource={selectedSource} t={t} />
    </div>
  );
}
