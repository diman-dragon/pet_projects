// components/popup/PopupHeader.tsx

import { X } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { LailaOrb } from '../orb/LailaOrb';
import { StatusDot } from '../shared/StatusDot';
import { IconButton } from '../../../../components/ui/atoms/IconButton';

interface PopupHeaderProps {
  isThinking: boolean;
  aiOnline: boolean | null;
  onClose: () => void;
}

/** Шапка виджета: орб, название, статус AI, кнопка закрытия */
export function PopupHeader({ isThinking, aiOnline, onClose }: PopupHeaderProps) {
  const { t } = useI18n();

  const dotColor = aiOnline === null ? 'bg-black/20' : aiOnline ? 'bg-emerald-400' : 'bg-red-400';

  const statusLabel =
    aiOnline === null
      ? t('laila.popup.statusInit')
      : aiOnline
        ? t('laila.popup.statusLocal')
        : t('laila.footer.localmindOffline');

  return (
    <div className="px-4 py-3 border-b border-black/[0.05] flex items-center justify-between bg-gray-50/50 shrink-0">
      <div className="flex items-center gap-2.5">
        <LailaOrb size="sm" isActive={isThinking} />
        <div className="flex flex-col">
          <span className="text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/70 leading-none">
            {t('laila.popup.brand')}
          </span>
          <div className="flex items-center gap-1.5 mt-0.5">
            <StatusDot colorClass={dotColor} pulse={aiOnline === true} />
            <span className="text-[length:var(--text-xs)] text-black/30 font-bold uppercase tracking-tighter">
              {statusLabel}
            </span>
          </div>
        </div>
      </div>

      <IconButton
        onClick={onClose}
        className="text-black/40 hover:text-black/80 p-2 hover:bg-black/5 rounded-full"
      >
        <X size={20} />
      </IconButton>
    </div>
  );
}
