// UPDATED: история через HistoryPanel (portal) + автоотправка при выборе из истории
import { RefObject, useCallback } from 'react';
import { Send, Square } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { IconButton } from '../../../../components/ui/atoms/IconButton';
import { useChatHistory } from '../../../../hooks/useChatHistory';
import { HistoryPanel, HistoryButton, useHistoryPanel } from '../../../../components/common/HistoryPanel';

interface PopupInputBarProps {
  inputRef: RefObject<HTMLInputElement | null>;
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onSendText: (text: string) => void; // автоотправка конкретного текста
  onAbort: () => void;
  isThinking: boolean;
}

export function PopupInputBar({
  inputRef,
  value,
  onChange,
  onSend,
  onSendText,
  onAbort,
  isThinking,
}: PopupInputBarProps) {
  const { t } = useI18n();
  const hp = useHistoryPanel();
  const { history, deleteOne, clearAll } = useChatHistory(hp.open);

  const historyItems = history.map((h) => ({ key: h.content, text: h.content }));

  const handleKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      isThinking ? onAbort() : onSend();
    }
  };

  // При выборе из истории — сразу отправляем, не просто вставляем
  const handleHistorySelect = useCallback(
    (text: string) => {
      if (isThinking) return;
      onSendText(text);
    },
    [isThinking, onSendText],
  );

  return (
    <div className="p-3 bg-white border-t border-black/[0.05] shrink-0">
      <div className="flex items-center gap-1.5 bg-gray-50 border border-black/[0.07] rounded-2xl p-1.5 focus-within:bg-white focus-within:border-[var(--app-accent)]/30 transition-all">
        {/* Кнопка истории */}
        <HistoryButton
          open={hp.open}
          onClick={hp.toggle}
          btnRef={hp.btnRef}
          hasItems={historyItems.length > 0}
          className="p-1.5"
        />

        <input
          ref={inputRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKey}
          placeholder={isThinking ? t('laila.popup.placeholderThinking') : t('laila.popup.placeholderIdle')}
          className="flex-1 bg-transparent text-[length:var(--text-sm)] px-1 py-1 text-black/70 placeholder:text-black/25 outline-none"
        />

        {isThinking ? (
          <IconButton
            onClick={onAbort}
            className="size-8 flex items-center justify-center rounded-xl bg-red-500 text-white hover:bg-red-600 shrink-0"
          >
            <Square size={14} fill="currentColor" />
          </IconButton>
        ) : (
          <IconButton
            onClick={onSend}
            disabled={!value.trim()}
            className="size-8 flex items-center justify-center rounded-xl bg-[var(--app-accent)] text-white hover:bg-[var(--app-accent-strong)] shrink-0 disabled:opacity-40"
          >
            <Send size={14} />
          </IconButton>
        )}
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
    </div>
  );
}
