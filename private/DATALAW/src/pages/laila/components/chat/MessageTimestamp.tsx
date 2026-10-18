// components/chat/MessageTimestamp.tsx

interface MessageTimestampProps {
  timestamp: Date;
  isUser: boolean;
}

/** Временная метка под сообщением */
export function MessageTimestamp({ timestamp, isUser }: MessageTimestampProps) {
  return (
    <div
      className={`text-[length:var(--text-xs)] mt-1.5 ${isUser ? 'text-white/50' : 'text-black/25'}`}
    >
      {timestamp.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })}
    </div>
  );
}
