import type { ReactNode } from 'react';
import { Sparkles, WifiOff } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

interface AIHeaderProps {
  streaming: boolean;
  offline: boolean;
  children?: ReactNode;
}

export function AIHeader({ streaming, offline, children }: AIHeaderProps) {
  const { t } = useI18n();

  return (
    <div className="mb-2.5 flex items-center gap-2">
      <div className="flex size-5 items-center justify-center rounded-md bg-[var(--app-accent)]/10 shrink-0">
        {offline ? (
          <WifiOff size={11} className="text-black/30" />
        ) : (
          <Sparkles
            size={11}
            className={`text-[var(--app-accent)] ${streaming ? 'animate-pulse' : ''}`}
          />
        )}
      </div>
      <span className="text-[length:var(--text-xs)] font-black uppercase tracking-wider text-[var(--app-accent)]/70">
        {t('knowledge.ai.headerLabel')}
      </span>
      {children}
    </div>
  );
}
