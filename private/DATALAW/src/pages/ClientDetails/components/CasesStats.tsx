// src/pages/ClientDetails/components/CasesStats.tsx
import { useMemo } from 'react';
import { Case } from '../../../types/entities';
import { Folder, PlayCircle, PauseCircle, CheckCircle2 } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { UiText } from '../../../components/ui';
import type { TranslationKey } from '../../../i18n/translations';

interface StatItem {
  key: 'total' | 'active' | 'paused' | 'done';
  labelKey: TranslationKey;
  val: number;
  icon: React.ReactNode;
  color: string;
}

interface Props {
  cases: Case[];
}

export function CasesStats({ cases }: Props) {
  const { t } = useI18n();

  const statsData = useMemo(() => {
    const counts = { total: cases.length, active: 0, paused: 0, done: 0 };

    cases.forEach((c) => {
      const s = (c.status || '').toUpperCase();
      if (s === 'ACTIVE') counts.active++;
      else if (s === 'PAUSED') counts.paused++;
      else if (s === 'DONE') counts.done++;
    });

    const items: StatItem[] = [
      {
        key: 'total',
        labelKey: 'cases.stats.total',
        val: counts.total,
        icon: <Folder size={14} />,
        color: 'text-black/40',
      },
      {
        key: 'active',
        labelKey: 'cases.status.ACTIVE',
        val: counts.active,
        icon: <PlayCircle size={14} />,
        color: 'text-[var(--app-accent)]',
      },
      {
        key: 'paused',
        labelKey: 'cases.status.PAUSED',
        val: counts.paused,
        icon: <PauseCircle size={14} />,
        color: 'text-amber-500',
      },
      {
        key: 'done',
        labelKey: 'cases.status.ARCHIVE',
        val: counts.done,
        icon: <CheckCircle2 size={14} />,
        color: 'text-green-600',
      },
    ];

    return items.filter((item) => item.val > 0 || item.key === 'total');
  }, [cases]);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 w-full">
      {statsData.map((s) => (
        <div
          key={s.key}
          className="group relative flex flex-col p-5 rounded-[var(--app-radius-lg)] bg-[var(--app-surface)] border border-[var(--app-border)] hover:border-[var(--app-border-strong)] hover:shadow-[var(--app-shadow-md)] transition-all duration-300 overflow-hidden"
        >
          <div className="flex items-center gap-3 mb-3">
             <div className={`p-2.5 rounded-xl bg-[var(--app-bg)]/50 border border-[var(--app-border)] ${s.color} transition-transform group-hover:scale-110 duration-500`}>
               {s.icon}
             </div>
             <UiText size="xs" weight="black" tone="muted" className="uppercase tracking-widest">
               {t(s.labelKey)}
             </UiText>
          </div>

          <div className="flex items-baseline gap-2">
            <UiText size="xl" weight="black" className="leading-none tracking-tight">
              {s.val}
            </UiText>
            <div className="size-1.5 rounded-full bg-[var(--app-border)] group-hover:bg-[var(--app-accent)] transition-colors mb-1" />
          </div>
        </div>
      ))}
    </div>
  );
}
