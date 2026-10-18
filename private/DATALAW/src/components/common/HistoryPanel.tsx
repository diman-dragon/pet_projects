import { Clock, History, Trash2, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export interface HistoryPanelItem {
  key: string;
  text: string;
}

interface HistoryPanelProps {
  items: HistoryPanelItem[];
  onSelect: (text: string) => void;
  onDeleteOne?: (text: string) => void;
  onClearAll?: () => void;
  label?: string;
  anchorRef: React.RefObject<HTMLElement | null>;
  open: boolean;
  onClose: () => void;
}

function usePortalPosition(anchorRef: React.RefObject<HTMLElement | null>, open: boolean) {
  const [pos, setPos] = useState({
    top: 0,
    left: 0,
    width: 0,
    placement: 'bottom' as 'bottom' | 'top',
  });

  const recalc = useCallback(() => {
    const el = anchorRef.current;
    if (!el) return;

    const rect = el.getBoundingClientRect();
    const panelHeight = 300;
    const margin = 8;
    const spaceBelow = window.innerHeight - rect.bottom - margin;
    const spaceAbove = rect.top - margin;

    if (spaceBelow >= panelHeight || spaceBelow >= spaceAbove) {
      setPos({
        top: rect.bottom + margin + window.scrollY,
        left: rect.left + window.scrollX,
        width: Math.max(rect.width, 280),
        placement: 'bottom',
      });
      return;
    }

    setPos({
      top: rect.top + window.scrollY - panelHeight - margin,
      left: rect.left + window.scrollX,
      width: Math.max(rect.width, 280),
      placement: 'top',
    });
  }, [anchorRef]);

  useEffect(() => {
    if (!open) return;

    recalc();
    window.addEventListener('scroll', recalc, true);
    window.addEventListener('resize', recalc);

    return () => {
      window.removeEventListener('scroll', recalc, true);
      window.removeEventListener('resize', recalc);
    };
  }, [open, recalc]);

  return pos;
}

export function HistoryPanel({
  items,
  onSelect,
  onDeleteOne,
  onClearAll,
  label = 'История',
  anchorRef,
  open,
  onClose,
}: HistoryPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const pos = usePortalPosition(anchorRef, open);

  useEffect(() => {
    if (!open) return;

    const handleClick = (event: MouseEvent) => {
      if (
        panelRef.current &&
        !panelRef.current.contains(event.target as Node) &&
        anchorRef.current &&
        !anchorRef.current.contains(event.target as Node)
      ) {
        onClose();
      }
    };

    const timer = setTimeout(() => document.addEventListener('mousedown', handleClick), 10);
    return () => {
      clearTimeout(timer);
      document.removeEventListener('mousedown', handleClick);
    };
  }, [anchorRef, onClose, open]);

  useEffect(() => {
    if (!open) return;

    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };

    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose, open]);

  if (!open) return null;

  const panel = (
    <div
      ref={panelRef}
      style={{
        position: 'absolute',
        top: pos.top,
        left: pos.left,
        width: pos.width,
        zIndex: 9999,
      }}
      className={[
        'bg-white rounded-2xl border border-black/[0.09]',
        'shadow-[0_8px_40px_-4px_rgba(0,0,0,0.18),0_2px_8px_rgba(0,0,0,0.06)]',
        'overflow-hidden',
        pos.placement === 'top'
          ? 'animate-in fade-in slide-in-from-bottom-2 duration-150'
          : 'animate-in fade-in slide-in-from-top-2 duration-150',
      ].join(' ')}
    >
      <div className="flex items-center justify-between border-b border-black/[0.06] bg-gray-50/80 px-3.5 py-2.5">
        <span className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-widest text-black/40">
          <Clock size={11} className="text-black/30" />
          {label}
        </span>
        <div className="flex items-center gap-0.5">
          {onClearAll && items.length > 0 && (
            <button
              type="button"
              onClick={onClearAll}
              className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-black/30 transition-all hover:bg-red-50 hover:text-red-400"
              title="Очистить всё"
            >
              <Trash2 size={10} />
              <span>Очистить</span>
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1 text-black/20 transition-all hover:bg-black/[0.05] hover:text-black/50"
          >
            <X size={13} />
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <div className="flex flex-col items-center justify-center px-4 py-8 text-center">
          <Clock size={22} className="mb-2 text-black/10" />
          <p className="text-[12px] text-black/25">Пока ничего нет</p>
        </div>
      ) : (
        <ul className="max-h-[240px] overflow-y-auto py-1.5">
          {items.map((item) => (
            <li key={item.key} className="group flex items-center px-1.5">
              <button
                type="button"
                onClick={() => {
                  onSelect(item.text);
                  onClose();
                }}
                className="flex min-w-0 flex-1 items-center gap-2.5 rounded-xl px-2.5 py-2 text-left transition-colors hover:bg-[var(--app-accent,_#6d28d9)]/[0.06]"
              >
                <Clock
                  size={12}
                  className="shrink-0 text-black/20 transition-colors group-hover:text-[var(--app-accent,_#6d28d9)]/40"
                />
                <span className="flex-1 truncate text-[13px] leading-snug text-black/65 transition-colors group-hover:text-black/85">
                  {item.text}
                </span>
              </button>

              {onDeleteOne && (
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    onDeleteOne(item.text);
                  }}
                  className="ml-0.5 shrink-0 rounded-lg p-1.5 text-black/20 opacity-0 transition-all group-hover:opacity-100 hover:bg-red-50 hover:text-red-400"
                  title="Удалить"
                >
                  <X size={12} />
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );

  return createPortal(panel, document.body);
}

interface HistoryButtonProps {
  open: boolean;
  onClick: () => void;
  btnRef: React.RefObject<HTMLButtonElement | null>;
  hasItems?: boolean;
  className?: string;
}

export function HistoryButton({
  open,
  onClick,
  btnRef,
  hasItems = true,
  className = '',
}: HistoryButtonProps) {
  return (
    <button
      ref={btnRef}
      type="button"
      onClick={onClick}
      title="История"
      className={[
        'flex items-center justify-center rounded-xl transition-all shrink-0',
        open
          ? 'text-[var(--app-accent,_#6d28d9)] bg-[var(--app-accent,_#6d28d9)]/10'
          : hasItems
            ? 'text-black/35 hover:text-black/60 hover:bg-black/[0.05]'
            : 'text-black/15 cursor-default',
        className,
      ].join(' ')}
    >
      <History size={15} />
    </button>
  );
}

export function useHistoryPanel() {
  const [open, setOpen] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);

  const toggle = useCallback(() => setOpen((value) => !value), []);
  const close = useCallback(() => setOpen(false), []);
  const show = useCallback(() => setOpen(true), []);

  return { open, toggle, close, show, btnRef };
}
