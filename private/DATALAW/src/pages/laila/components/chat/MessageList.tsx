// components/chat/MessageList.tsx

import { RefObject } from 'react';
import { ChatMessage } from '../../types';
import { MessageBubble } from './MessageBubble';

interface MessageListProps {
  messages: ChatMessage[];
  bottomRef: RefObject<HTMLDivElement | null>;
}

/** Скроллируемый список сообщений чата */
export function MessageList({ messages, bottomRef }: MessageListProps) {
  return (
    <div className="flex-1 overflow-y-auto px-6 py-5 space-y-4 custom-scroll">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} msg={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
