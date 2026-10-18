// src/pages/laila/components/chat/ChatSessionsDrawer.tsx
// UPDATED: рендер через portal, красивый дизайн, удаление по одному и всего

import { createPortal } from 'react-dom';
import { useEffect, useRef } from 'react';
import { X, MessageSquare, Trash2, Clock, ChevronRight, Loader2, History } from 'lucide-react';
import { useChatSessions } from '../../../../hooks/useChatSessions';
import type { ChatMessage } from '../../types';

interface ChatSessionsDrawerProps {
  open: boolean;
  onClose: () => void;
  currentSessionId: string;
  onLoadSession: (messages: ChatMessage[]) => void;
  /** ref кнопки-триггера для расчёта позиции drawer */
  triggerRef?: React.RefObject<HTMLElement | null>;
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return '';
  try {
    const d = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffDays = Math.floor(diffMs / 86400000);
    if (diffDays === 0) return d.toLocaleTimeString('ru', { hour: '2-digit', minute: '2-digit' });
    if (diffDays === 1) return 'Вчера';
    if (diffDays < 7) return `${diffDays} дн.`;
    return d.toLocaleDateString('ru', { day: 'numeric', month: 'short' });
  } catch { return ''; }
}

function getSessionTitle(session: { title: string | null; preview: string }): string {
  if (session.title?.trim()) return session.title;
  if (session.preview) return session.preview.length > 55 ? session.preview.slice(0, 55) + '…' : session.preview;
  return 'Без названия';
}

function DrawerContent({
  open,
  onClose,
  currentSessionId,
  onLoadSession,
}: ChatSessionsDrawerProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const { sessions, loading, loadMessages, deleteOne, clearAll } = useChatSessions(open, 'chat');

  // Закрыть по Escape
  useEffect(() => {
    if (!open) return;
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [open, onClose]);

  // Закрыть по клику снаружи
  useEffect(() => {
    if (!open) return;
    const timer = setTimeout(() => {
      const handleClick = (e: MouseEvent) => {
        if (panelRef.current && !panelRef.current.contains(e.target as Node)) onClose();
      };
      document.addEventListener('mousedown', handleClick);
      return () => document.removeEventListener('mousedown', handleClick);
    }, 50);
    return () => clearTimeout(timer);
  }, [open, onClose]);

  const handleLoad = async (sessionId: string) => {
    if (sessionId === currentSessionId) { onClose(); return; }
    const records = await loadMessages(sessionId);
    const chatMessages: ChatMessage[] = records.map((r) => ({
      id: r.id,
      role: r.role === 'user' ? 'user' : 'laila',
      text: r.content,
      timestamp: new Date(r.created_at),
    }));
    if (chatMessages.length > 0) onLoadSession(chatMessages);
    onClose();
  };

  if (!open) return null;

  const drawer = (
    <div style={{ position: 'fixed', inset: 0, zIndex: 9990, pointerEvents: 'none' }}>
      {/* Backdrop */}
      <div
        style={{ pointerEvents: 'auto' }}
        className="absolute inset-0 bg-black/10 backdrop-blur-[2px] animate-in fade-in duration-150"
        onClick={onClose}
      />

      {/* Panel */}
      <div
        ref={panelRef}
        style={{ pointerEvents: 'auto' }}
        className="absolute top-0 left-0 bottom-0 w-[300px] bg-white border-r border-black/[0.07] flex flex-col shadow-[6px_0_40px_rgba(0,0,0,0.12)] animate-in slide-in-from-left duration-200"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3.5 border-b border-black/[0.06] bg-gradient-to-r from-gray-50 to-white">
          <div className="flex items-center gap-2">
            <div className="size-7 rounded-lg bg-[var(--app-accent)]/10 flex items-center justify-center">
              <History size={14} className="text-[var(--app-accent)]" />
            </div>
            <div>
              <p className="text-[13px] font-semibold text-black/80">История чатов</p>
              {sessions.length > 0 && (
                <p className="text-[11px] text-black/30">{sessions.length} сессий</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-0.5">
            {sessions.length > 0 && (
              <button
                type="button"
                onClick={clearAll}
                title="Удалить все чаты"
                className="p-1.5 rounded-xl text-black/25 hover:text-red-400 hover:bg-red-50 transition-all"
              >
                <Trash2 size={13} />
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="p-1.5 rounded-xl text-black/25 hover:text-black/50 hover:bg-black/[0.04] transition-all"
            >
              <X size={15} />
            </button>
          </div>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto py-2" style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(0,0,0,0.1) transparent' }}>
          {loading && (
            <div className="flex items-center justify-center py-10">
              <Loader2 size={20} className="animate-spin text-black/20" />
            </div>
          )}

          {!loading && sessions.length === 0 && (
            <div className="flex flex-col items-center justify-center py-14 px-6 text-center">
              <div className="size-12 rounded-2xl bg-black/[0.04] flex items-center justify-center mb-3">
                <MessageSquare size={20} className="text-black/20" />
              </div>
              <p className="text-[13px] font-medium text-black/30">Нет сохранённых чатов</p>
              <p className="text-[11px] text-black/20 mt-1">Начните новый диалог</p>
            </div>
          )}

          {!loading && sessions.map((session) => {
            const isCurrent = session.id === currentSessionId;
            return (
              <div key={session.id} className="px-2 mb-0.5">
                <div
                  onClick={() => handleLoad(session.id)}
                  className={[
                    'group flex items-start gap-2.5 p-3 rounded-xl cursor-pointer transition-all',
                    isCurrent
                      ? 'bg-[var(--app-accent)]/8 ring-1 ring-[var(--app-accent)]/15'
                      : 'hover:bg-black/[0.03]',
                  ].join(' ')}
                >
                  {/* Icon */}
                  <div className={[
                    'size-8 rounded-xl flex items-center justify-center shrink-0 mt-0.5',
                    isCurrent ? 'bg-[var(--app-accent)]/15' : 'bg-black/[0.05] group-hover:bg-black/[0.08]',
                  ].join(' ')}>
                    <MessageSquare
                      size={13}
                      className={isCurrent ? 'text-[var(--app-accent)]' : 'text-black/30'}
                    />
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <p className={[
                      'text-[13px] font-medium truncate leading-snug',
                      isCurrent ? 'text-[var(--app-accent)]' : 'text-black/70 group-hover:text-black/90',
                    ].join(' ')}>
                      {getSessionTitle(session)}
                    </p>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <Clock size={10} className="text-black/20 shrink-0" />
                      <span className="text-[11px] text-black/30">
                        {formatDate(session.lastMessageAt ?? session.createdAt)}
                      </span>
                      <span className="text-[10px] text-black/15">· {session.messageCount}</span>
                    </div>
                    {isCurrent && (
                      <span className="inline-block mt-1 text-[10px] font-medium text-[var(--app-accent)]/60 bg-[var(--app-accent)]/8 px-1.5 py-0.5 rounded-md">
                        текущий
                      </span>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex flex-col items-end gap-1 shrink-0">
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); deleteOne(session.id); }}
                      className="p-1 rounded-lg opacity-0 group-hover:opacity-100 text-black/20 hover:text-red-400 hover:bg-red-50 transition-all"
                      title="Удалить"
                    >
                      <Trash2 size={11} />
                    </button>
                    {!isCurrent && (
                      <ChevronRight size={12} className="opacity-0 group-hover:opacity-30 text-black/50 mt-1" />
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );

  return createPortal(drawer, document.body);
}

export function ChatSessionsDrawer(props: ChatSessionsDrawerProps) {
  if (typeof document === 'undefined') return null;
  return <DrawerContent {...props} />;
}
