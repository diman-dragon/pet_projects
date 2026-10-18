import { useI18n } from '../../../i18n/AppLanguageProvider';
// src/components/ui/layout/Window.tsx
import { ReactNode } from 'react';
import { X } from 'lucide-react';

interface WindowProps {
  children: ReactNode;
  title?: string;
  footer?: ReactNode;
  onClose?: () => void;
  className?: string;
  scrollable?: boolean;
  noPadding?: boolean;
  /** Режим "страницы" — без шапки, скруглений и декоративных эффектов */
  pageMode?: boolean;
}

export function Window({
  children,
  title,
  footer,
  onClose,
  className = '',
  scrollable = true,
  noPadding = false,
  pageMode = false,
}: WindowProps) {
  const { t } = useI18n();
  // PageMode для PageBase — чистый контейнер без украшений
  if (pageMode) {
    return (
      <div
        className={[
          'flex-1 flex flex-col bg-white overflow-hidden',
          scrollable ? 'overflow-y-auto custom-scrollbar' : 'overflow-hidden',
          className,
        ]
          .filter(Boolean)
          .join(' ')}
      >
        <div className={noPadding ? '' : 'p-6'}>{children}</div>
      </div>
    );
  }

  // Оригинальный режим для модальных окон, диалогов и т.д.
  return (
    <div
      className={[
        'flex flex-col h-full mica-effect rounded-[12px] overflow-hidden',
        'shadow-[0_20px_50px_rgba(0,0,0,0.15),0_1px_1px_rgba(255,255,255,0.5)_inset]',
        className,
      ].join(' ')}
    >
      {/* ШАПКА ОКНА — показываем только если есть title ИЛИ onClose */}
      {(title || onClose) && (
        <div className="shrink-0 h-[52px] px-6 flex items-center justify-between bg-black/[0.02] border-b border-black/[0.05]">
          {title ? (
            <h2 className="text-[14px] font-semibold text-black/80 tracking-tight select-none">
              {title}
            </h2>
          ) : (
            <div />
          )}

          {onClose && (
            <button
              onClick={onClose}
              className={[
                'w-8 h-8 -mr-2 flex items-center justify-center rounded-md transition-all duration-200',
                'text-black/40 hover:text-white hover:bg-[#c42b1c] active:bg-[#a82317]',
              ].join(' ')}
              aria-label={t('common.closeAria')}
            >
              <X size={16} strokeWidth={2.5} />
            </button>
          )}
        </div>
      )}

      {/* КОНТЕНТ */}
      <div
        className={[
          'flex-1 relative',
          scrollable ? 'overflow-y-auto custom-scrollbar' : 'overflow-hidden',
        ].join(' ')}
      >
        <div className={noPadding ? 'h-full' : 'p-6'}>{children}</div>
      </div>

      {/* ПОДВАЛ */}
      {footer && (
        <div className="shrink-0 px-6 py-4 border-t border-black/[0.05] bg-[#f9f9f9]/80 backdrop-blur-sm">
          <div className="flex items-center justify-end gap-2">{footer}</div>
        </div>
      )}
    </div>
  );
}
