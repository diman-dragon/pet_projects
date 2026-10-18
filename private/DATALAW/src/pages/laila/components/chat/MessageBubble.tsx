// components/chat/MessageBubble.tsx

import { ChatMessage } from '../../types';
import { LailaOrb } from '../orb/LailaOrb';
import { UserAvatar } from './UserAvatar';
import { MessageTimestamp } from './MessageTimestamp';
import { TypingDots } from './TypingDots';
import { StreamingCursor } from '../../../../components/ui/feedback/StreamingCursor';
import { MarkdownMessage } from '../shared/MarkdownMessage';

interface MessageBubbleProps {
  msg: ChatMessage;
}

/** Одно сообщение в чате: аватар + пузырь + временная метка */
export function MessageBubble({ msg }: MessageBubbleProps) {
  const isUser = msg.role === 'user';

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      {isUser ? <UserAvatar /> : <LailaOrb size="sm" isActive={msg.isTyping} />}

      {/* Bubble */}
      <div
        className={`max-w-[75%] px-4 py-3 rounded-2xl text-[length:var(--text-sm)] leading-[var(--leading-body)]
          ${isUser
            ? 'bg-[var(--app-accent)] text-white rounded-tr-sm shadow-sm shadow-[#0067c0]/20'
            : 'bg-black/[0.03] text-black/75 rounded-tl-sm border border-black/[0.05]'
          }`}
      >
        {/* Empty streaming state → dots */}
        {!isUser && msg.isTyping && !msg.text ? (
          <TypingDots />
        ) : (
          <>
            <MarkdownMessage content={msg.text} isUser={isUser} />
            {msg.isTyping && <StreamingCursor />}
          </>
        )}

        <MessageTimestamp timestamp={msg.timestamp} isUser={isUser} />
      </div>
    </div>
  );
}
