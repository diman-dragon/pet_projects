import { useCallback } from 'react';
import { HistoryPanel, HistoryButton, useHistoryPanel } from '../../../../components/common/HistoryPanel';
import { UiText } from '../../../../components/ui/atoms/UiText';
import { useChatHistory } from '../../../../hooks/useChatHistory';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { ChatInputActions } from './ChatInputActions';

interface ChatInputBarProps {
  value: string;
  onChange: (v: string) => void;
  onSend: (text?: string) => void;
  onClear: () => void;
  onStop?: () => void;
  isThinking: boolean;
}

export function ChatInputBar({
  value,
  onChange,
  onSend,
  onClear,
  onStop,
  isThinking,
}: ChatInputBarProps) {
  const { t } = useI18n();
  const hp = useHistoryPanel();
  const { history, deleteOne, clearAll } = useChatHistory(hp.open);

  const historyItems = history.map((item) => ({ key: item.content, text: item.content }));

  const handleKey = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  };

  const handleHistorySelect = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      onChange(text);
      onSend(text);
    },
    [onChange, onSend],
  );

  return (
    <div className="px-4 pb-4 shrink-0">
      <div className="flex items-end gap-2 rounded-2xl border border-black/[0.07] bg-black/[0.02] p-2 transition-all focus-within:border-violet-400/40 focus-within:ring-2 focus-within:ring-violet-400/10">
        <HistoryButton
          open={hp.open}
          onClick={hp.toggle}
          btnRef={hp.btnRef}
          hasItems={historyItems.length > 0}
          className="mb-0.5 p-2"
        />

        <textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKey}
          placeholder={t('laila.chat.placeholder')}
          rows={1}
          className="max-h-[120px] flex-1 resize-none bg-transparent px-1 py-1 text-[length:var(--text-sm)] leading-[var(--leading-body)] text-black/70 outline-none placeholder:text-black/30"
          style={{ minHeight: '36px' }}
        />

        <ChatInputActions
          isThinking={isThinking}
          canSend={!!value.trim()}
          onSend={() => onSend()}
          onStop={onStop}
          onClear={onClear}
        />
      </div>

      <HistoryPanel
        items={historyItems}
        onSelect={handleHistorySelect}
        onDeleteOne={deleteOne}
        onClearAll={clearAll}
        label="Недавние сообщения"
        anchorRef={hp.btnRef as React.RefObject<HTMLElement | null>}
        open={hp.open}
        onClose={hp.close}
      />

      <UiText as="p" size="xs" weight="medium" tone="muted" className="mt-2 text-center text-black/20">
        {t('laila.chat.disclaimer')}
      </UiText>
    </div>
  );
}
