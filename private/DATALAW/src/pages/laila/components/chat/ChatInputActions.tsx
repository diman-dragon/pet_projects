// components/chat/ChatInputActions.tsx

import { Send, Square, Trash2 } from 'lucide-react';
import { IconButton } from '../../../../components/ui/atoms/IconButton';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

interface ChatInputActionsProps {
  isThinking: boolean;
  canSend: boolean;
  onSend: () => void;
  onStop?: () => void;
  onClear: () => void;
}

/** Кнопки действий справа в поле ввода: очистить / остановить / отправить */
export function ChatInputActions({
  isThinking,
  canSend,
  onSend,
  onStop,
  onClear,
}: ChatInputActionsProps) {
  const { t } = useI18n();
  return (
    <div className="flex items-center gap-1 shrink-0">
      {/* Очистить чат */}
      <IconButton
        onClick={onClear}
        title={t('laila.chat.clearTitle')}
        className="p-2 rounded-xl text-black/25 hover:text-black/50 hover:bg-black/[0.04]"
      >
        <Trash2 size={13} />
      </IconButton>

      {/* Стоп / Отправить */}
      {isThinking ? (
        <IconButton
          onClick={onStop ?? (() => {})}
          title={t('laila.chat.stopTitle')}
          className="p-2 rounded-xl bg-violet-500 text-white hover:bg-violet-600 shadow-sm shadow-violet-500/20"
        >
          <Square size={14} />
        </IconButton>
      ) : (
        <IconButton
          onClick={onSend}
          disabled={!canSend}
          className="p-2 rounded-xl bg-[var(--app-accent)] text-white hover:bg-[var(--app-accent-strong)] shadow-sm shadow-[#0067c0]/20"
        >
          <Send size={14} />
        </IconButton>
      )}
    </div>
  );
}
