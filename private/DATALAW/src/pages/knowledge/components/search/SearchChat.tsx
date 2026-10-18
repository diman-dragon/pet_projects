// src/pages/knowledge/components/search/SearchChat.tsx
// Чат поиска - история вопросов и ответов с фрагментами

import { useEffect, useRef, useState } from 'react';
import { Send, Sparkles } from 'lucide-react';
import { MarkdownMessage } from '../../../laila/components/shared/MarkdownMessage';
import { StreamingCursor } from '../../../../components/ui/feedback/StreamingCursor';

interface SearchChatProps {
  query: string;
  onQueryChange: (query: string) => void;
  onSearch: (query: string) => void;
  isRunning: boolean;
  currentPhase: 'idle' | 'running' | 'done' | 'error';
  currentAnswer: string;
  isAnalyzing: boolean;
  onSelectLink?: (href: string) => boolean | void;
  onAbort?: () => void;
}

export function SearchChat({
  query,
  onQueryChange,
  onSearch,
  isRunning,
  currentPhase,
  currentAnswer,
  isAnalyzing,
  onSelectLink,
}: SearchChatProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [inputValue, setInputValue] = useState('');

  useEffect(() => {
    setInputValue(query);
  }, [query]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [currentAnswer, isAnalyzing, query]);

  const handleInputChange = (newValue: string) => {
    setInputValue(newValue);
    onQueryChange(newValue);
  };

  const handleSubmit = (e?: React.FormEvent | React.KeyboardEvent) => {
    if (e) e.preventDefault();
    if (inputValue.trim() && !isRunning) {
      onSearch(inputValue);
    }
  };

  return (
    <div className="flex h-full flex-col overflow-hidden bg-white">
      <div ref={scrollRef} className="custom-scroll flex-1 overflow-y-auto p-4 space-y-4">
        {currentPhase === 'idle' ? (
          <div className="flex h-full items-center justify-center text-center">
            <div>
              <div className="mb-3 flex justify-center">
                <Sparkles size={24} className="text-black/20" />
              </div>
              <p className="text-[13px] text-black/40">Чат Laila готов ответить на ваши вопросы</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {query && (
              <div className="flex justify-end">
                <div className="max-w-[75%] rounded-3xl border border-[var(--app-accent)]/15 bg-[var(--app-accent)]/10 px-4 py-3 text-[13px] text-black/85 shadow-sm">
                  {query}
                </div>
              </div>
            )}

            <div className="flex gap-3">
              <div className="mt-1 shrink-0 rounded-full bg-black/[0.05] p-3">
                <Sparkles size={18} className="text-[var(--app-accent)]" />
              </div>
              <div className="max-w-[75%] rounded-3xl border border-black/[0.08] bg-black/[0.02] px-4 py-4 text-[13px] text-black/80 shadow-sm">
                {currentAnswer ? (
                  <>
                    <MarkdownMessage content={currentAnswer} onLinkClick={onSelectLink} />
                    {isAnalyzing && <StreamingCursor />}
                  </>
                ) : (
                  <div className="flex gap-2 text-black/40">
                    <span className="inline-block h-2.5 w-2.5 rounded-full animate-bounce bg-black/30" />
                    <span className="inline-block h-2.5 w-2.5 rounded-full animate-bounce bg-black/30" style={{ animationDelay: '120ms' }} />
                    <span className="inline-block h-2.5 w-2.5 rounded-full animate-bounce bg-black/30" style={{ animationDelay: '240ms' }} />
                    <span>Генерация ответа…</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="shrink-0 border-t border-black/[0.06] bg-white p-3">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            placeholder="Спросите Laila что-нибудь..."
            value={inputValue}
            onChange={(e) => handleInputChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !isRunning) {
                handleSubmit(e);
              }
            }}
            disabled={isRunning}
            className="flex-1 rounded-full border border-black/[0.06] bg-white px-4 py-3 text-[13px] placeholder-black/30 outline-none transition focus:border-[var(--app-accent)]/30 focus:ring-1 focus:ring-[var(--app-accent)]/20 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isRunning}
            className="flex h-11 w-11 items-center justify-center rounded-full bg-[var(--app-accent)] text-white transition hover:opacity-90 disabled:opacity-50"
          >
            <Send size={16} />
          </button>
        </div>
      </form>
    </div>
  );
}
