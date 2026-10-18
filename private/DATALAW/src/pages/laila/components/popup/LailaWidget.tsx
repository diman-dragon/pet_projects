// src/pages/laila/components/popup/LailaWidget.tsx
//
// FIX: виджет теперь использует useChatStream вместо самописного handleSend в LailaPopup.
// Это означает что все диалоги через виджет-FAB пишутся в БД и участвуют
// в памяти Лайлы (STM / MTM / LTM) — точно так же как полноэкранный чат.

import { useState, useMemo } from 'react';
import { LailaOrb } from '../orb/LailaOrb';
import { LailaPopup } from './LailaPopup';
import { useChatStream } from '../../hooks/useChatStream';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { WIDGET_HISTORY_LIMIT } from '../../../../services/ai/config';

export function LailaWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const { t } = useI18n();

  const welcomeMessage = useMemo(
    () => ({
      id: 'welcome',
      role: 'laila' as const,
      text: t('laila.chat.welcome'),
      timestamp: new Date(),
    }),
    [t],
  );

  const { messages, isThinking, send, stop } = useChatStream([welcomeMessage], {
    modelAgentId: 'widget',
    assistantId: 'laila',
    origin: 'widget',
    contextScope: 'laila:widget',
    historyLimit: WIDGET_HISTORY_LIMIT,
    sessionStorageKey: 'Dorlane:laila:session:widget',
  });

  return (
    <>
      <button
        onClick={() => setIsOpen((v) => !v)}
        aria-label={t('laila.widget.open')}
        className={`
          size-12 rounded-full flex items-center justify-center
          bg-white border border-black/[0.08]
          shadow-[0_8px_32px_rgba(0,0,0,0.14)]
          transition-all duration-300
          ${isOpen ? 'scale-95 shadow-md' : 'hover:shadow-[0_12px_40px_rgba(0,0,0,0.18)] hover:scale-105'}
        `}
      >
        <LailaOrb size="fab" isActive={isThinking} />
      </button>

      {isOpen && (
        <LailaPopup
          messages={messages}
          isThinking={isThinking}
          onSend={send}
          onAbort={stop}
          onClose={() => setIsOpen(false)}
        />
      )}
    </>
  );
}
