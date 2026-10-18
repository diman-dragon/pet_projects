// UPDATED: история через HistoryPanel (portal) + кнопка-триггер
import { Loader2, Send, Square } from 'lucide-react';
import type { TranslationKey } from '../../../../i18n/translations';
import { useSearchHistory } from '../../../../hooks/useSearchHistory';
import { HistoryPanel, HistoryButton, useHistoryPanel } from '../../../../components/common/HistoryPanel';

interface SemanticComposerProps {
  input: string;
  isStreaming: boolean;
  isThinking: boolean;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
  onInputChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onSubmit: () => void;
  onAbort: () => void;
  t: (key: TranslationKey) => string;
}

export function SemanticComposer({
  input,
  isStreaming,
  isThinking,
  inputRef,
  onInputChange,
  onKeyDown,
  onSubmit,
  onAbort,
  t,
}: SemanticComposerProps) {
  const hp = useHistoryPanel();
  const { history, deleteOne, clearAll } = useSearchHistory(hp.open);

  const historyItems = history.map((h) => ({ key: h.query, text: h.query }));

  const handleHistorySelect = (text: string) => {
    const ev = { target: { value: text, style: {} } } as unknown as React.ChangeEvent<HTMLTextAreaElement>;
    onInputChange(ev);
    inputRef.current?.focus();
  };

  return (
    <div className="border-t border-black/[0.06] p-4">
      <div className="mx-auto max-w-3xl">
        <div className="relative rounded-2xl border border-black/[0.08] bg-white shadow-sm">
          <textarea
            ref={inputRef}
            value={input}
            onChange={onInputChange}
            onKeyDown={onKeyDown}
            placeholder={isThinking ? t('knowledge.semantic.thinkingPlaceholder') : t('knowledge.semantic.inputPlaceholder')}
            disabled={isStreaming || isThinking}
            rows={1}
            className="w-full resize-none bg-transparent px-4 py-3.5 pr-20 text-sm text-black/80 outline-none placeholder:text-black/40"
          />

          {/* Правая часть: кнопка истории + отправить/стоп */}
          <div className="absolute bottom-2 right-2 flex items-center gap-1">
            <HistoryButton
              open={hp.open}
              onClick={hp.toggle}
              btnRef={hp.btnRef}
              hasItems={historyItems.length > 0}
              className="p-2"
            />
            {isThinking ? (
              <div className="p-2">
                <Loader2 className="h-4 w-4 animate-spin text-[var(--app-accent)]" />
              </div>
            ) : isStreaming ? (
              <button type="button" onClick={onAbort} className="rounded-lg bg-red-50 p-2 text-red-500 hover:bg-red-100 transition-colors">
                <Square className="h-4 w-4" />
              </button>
            ) : (
              <button type="button" onClick={onSubmit} disabled={!input.trim()} className="rounded-lg bg-[var(--app-accent)] p-2 text-white hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-opacity">
                <Send className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

        <HistoryPanel
          items={historyItems}
          onSelect={handleHistorySelect}
          onDeleteOne={deleteOne}
          onClearAll={clearAll}
          label="Недавние запросы"
          anchorRef={hp.btnRef as React.RefObject<HTMLElement | null>}
          open={hp.open}
          onClose={hp.close}
        />

        <p className="mt-2 text-center text-[11px] text-black/40">
          {t('knowledge.semantic.groundedHint')}
        </p>
      </div>
    </div>
  );
}
