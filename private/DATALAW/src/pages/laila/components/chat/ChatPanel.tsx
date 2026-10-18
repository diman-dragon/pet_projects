// components/chat/ChatPanel.tsx
// UPDATED: добавлена кнопка и боковая панель истории чатов

import { RefObject, useMemo, useState, useCallback } from 'react';
import { History } from 'lucide-react';
import { ChatMessage } from '../../types';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../../i18n/translations';
import { MessageList } from './MessageList';
import { QuickPrompts } from './QuickPrompts';
import { ChatInputBar } from './ChatInputBar';
import { ChatSessionsDrawer } from './ChatSessionsDrawer';

interface ChatPanelProps {
  messages: ChatMessage[];
  input: string;
  onInputChange: (v: string) => void;
  onSend: (text?: string) => void;
  onClear: () => void;
  onStop?: () => void;
  isThinking: boolean;
  bottomRef: RefObject<HTMLDivElement | null>;
  sessionId: string;
  onLoadSession: (messages: ChatMessage[]) => void;
}

const QUICK_KEYS: TranslationKey[] = [
  'laila.quickPrompt.0',
  'laila.quickPrompt.1',
  'laila.quickPrompt.2',
  'laila.quickPrompt.3',
];

export function ChatPanel({
  messages,
  input,
  onInputChange,
  onSend,
  onClear,
  onStop,
  isThinking,
  bottomRef,
  sessionId,
  onLoadSession,
}: ChatPanelProps) {
  const { t } = useI18n();
  const quickPrompts = useMemo(() => QUICK_KEYS.map((k) => t(k)), [t]);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const handleLoadSession = useCallback(
    (msgs: ChatMessage[]) => {
      onLoadSession(msgs);
    },
    [onLoadSession],
  );

  return (
    <div className="relative flex flex-col h-full overflow-hidden">
      {/* Toolbar: кнопка истории чатов */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-black/[0.04] shrink-0">
        <button
          type="button"
          onClick={() => setDrawerOpen((v) => !v)}
          className={[
            'flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[length:var(--text-xs)]',
            'border transition-colors',
            drawerOpen
              ? 'border-[var(--app-accent)]/25 bg-[var(--app-accent)]/8 text-[var(--app-accent)]'
              : 'border-black/[0.07] bg-transparent text-black/35 hover:text-black/55 hover:bg-black/[0.03]',
          ].join(' ')}
          title="История чатов"
        >
          <History size={13} />
          <span className="hidden sm:inline">История чатов</span>
        </button>
      </div>

      {/* Sessions drawer */}
      <ChatSessionsDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        currentSessionId={sessionId}
        onLoadSession={handleLoadSession}

      />

      <MessageList messages={messages} bottomRef={bottomRef} />

      {messages.length <= 1 && <QuickPrompts prompts={quickPrompts} onSelect={onSend} />}

      <ChatInputBar
        value={input}
        onChange={onInputChange}
        onSend={onSend}
        onClear={onClear}
        onStop={onStop}
        isThinking={isThinking}
      />
    </div>
  );
}
