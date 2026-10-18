// src/pages/laila/components/popup/PopupMessageList.tsx

import { RefObject } from 'react';
import { ChatMessage } from '../../types';
import { MarkdownMessage } from '../shared/MarkdownMessage';
import { LailaThinkingAnimation } from './LailaThinkingAnimation';

interface PopupMessageListProps {
  messages: ChatMessage[];
  isThinking: boolean;
  bottomRef: RefObject<HTMLDivElement | null>;
}

export function PopupMessageList({ messages, isThinking, bottomRef }: PopupMessageListProps) {
  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 scrollbar-hide select-text">
      {messages.map((m) => (
        <div key={m.id} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
          <div
            className={`max-w-[85%] px-3 py-2.5 rounded-2xl shadow-sm ${
              m.role === 'user'
                ? 'bg-[var(--app-accent)] text-white rounded-tr-none'
                : 'bg-[var(--app-surface-soft)] text-black/70 rounded-tl-none border border-black/[0.03]'
            }`}
          >
            <MarkdownMessage content={m.text} isUser={m.role === 'user'} />
          </div>
        </div>
      ))}
      {isThinking && <LailaThinkingAnimation />}
      <div ref={bottomRef} />
    </div>
  );
}
