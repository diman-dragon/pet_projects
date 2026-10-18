// src/pages/laila/components/popup/LailaPopup.tsx
//
// FIX: вся логика отправки и стриминга удалена отсюда.
// Теперь попап получает готовые send/onAbort из LailaWidget,
// который держит useChatStream. Это обеспечивает запись диалогов в БД.

import { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { ChatMessage } from '../../types';
import { useOllamaStore } from '../../../../store/ollamaStore';
import { PopupHeader } from './PopupHeader';
import { PopupOfflineBanner } from './PopupOfflineBanner';
import { PopupMessageList } from './PopupMessageList';
import { PopupInputBar } from './PopupInputBar';

interface LailaPopupProps {
  messages: ChatMessage[];
  isThinking: boolean;
  onSend: (value: string) => void;
  onAbort: () => void;
  onClose: () => void;
}

export function LailaPopup({ messages, isThinking, onSend, onAbort, onClose }: LailaPopupProps) {
  const [input, setInput] = useState('');
  const { sidecarStatus, refreshStatus } = useOllamaStore();
  const aiOnline = sidecarStatus.running;

  const popupRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // ── Autofocus ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (isThinking) return;
    const t = setTimeout(() => inputRef.current?.focus(), 180);
    return () => clearTimeout(t);
  }, [isThinking]);

  // ── Initial scroll + AI status check ─────────────────────────────────────────
  useEffect(() => {
    refreshStatus();
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'instant' }), 120);
  }, [refreshStatus]);

  // ── Auto-scroll on new messages ───────────────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isThinking]);

  // ── Close on outside click ────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const t = setTimeout(() => document.addEventListener('mousedown', handler), 100);
    return () => {
      clearTimeout(t);
      document.removeEventListener('mousedown', handler);
    };
  }, [onClose]);

  // ── Send ──────────────────────────────────────────────────────────────────
  const handleSend = useCallback(() => {
    const value = input.trim();
    if (!value || isThinking) return;
    setInput('');
    onSend(value);
  }, [input, isThinking, onSend]);

  // Автоотправка конкретного текста (из истории)
  const handleSendText = useCallback((text: string) => {
    const value = text.trim();
    if (!value || isThinking) return;
    setInput('');
    onSend(value);
  }, [isThinking, onSend]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return createPortal(
    <div
      ref={popupRef}
      className="fixed bottom-20 right-6 w-[340px] h-[480px] bg-white rounded-[24px] shadow-[0_20px_60px_rgba(0,0,0,0.18)] border border-black/[0.08] flex flex-col overflow-hidden z-[9999] animate-in fade-in zoom-in duration-200 origin-bottom-right"
      onClick={(e) => e.stopPropagation()}
    >
      <PopupHeader isThinking={isThinking} aiOnline={aiOnline} onClose={onClose} />

      {aiOnline === false && <PopupOfflineBanner />}

      <PopupMessageList messages={messages} isThinking={isThinking} bottomRef={bottomRef} />

      <PopupInputBar
        inputRef={inputRef}
        value={input}
        onChange={setInput}
        onSend={handleSend}
        onSendText={handleSendText}
        onAbort={onAbort}
        isThinking={isThinking}
      />
    </div>,
    document.body,
  );
}
