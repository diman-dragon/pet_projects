import React, { useState } from 'react';
import { Info, ChevronDown, X } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { InfoCards } from '../InfoCards';
import type { useAppealDetailsLogic } from '../../useAppealDetailsLogic';

interface AppealInfoDropdownProps {
  appeal: NonNullable<ReturnType<typeof useAppealDetailsLogic>['appeal']>;
}

export const AppealInfoDropdown: React.FC<AppealInfoDropdownProps> = ({ appeal }) => {
  const { t, language } = useI18n();
  const [open, setOpen] = useState(false);

  const dateLocale = language === 'sr' ? 'sr-RS' : language === 'en' ? 'en-US' : 'ru-RU';
  const createdAt = appeal.created_at
    ? new Date(appeal.created_at).toLocaleDateString(dateLocale, {
        day: '2-digit',
        month: '2-digit',
        year: '2-digit',
      })
    : '—';

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="h-7 flex items-center gap-1.5 px-2.5 rounded-md border border-black/[0.08] text-[length:var(--text-xs)] text-black/60 font-medium hover:bg-black/[0.04] transition-all"
      >
        <Info size={12} className="text-black/30" />
        <span>{createdAt}</span>
        <ChevronDown
          size={11}
          className={`text-black/30 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-9 z-40 w-[260px] bg-white border border-black/[0.08] rounded-xl shadow-xl p-4 animate-in fade-in slide-in-from-top-1 duration-150">
            <div className="flex items-center justify-between mb-3">
              <span className="text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/30">
                {t('appeal.info')}
              </span>
              <button
                onClick={() => setOpen(false)}
                className="text-black/25 hover:text-black/50 transition-colors"
              >
                <X size={12} />
              </button>
            </div>
            <InfoCards appeal={appeal} />
          </div>
        </>
      )}
    </div>
  );
};
