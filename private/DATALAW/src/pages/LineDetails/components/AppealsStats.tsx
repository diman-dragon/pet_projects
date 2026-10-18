// src/pages/LineDetails/components/AppealsStats.tsx
import { Appeal } from '../../../types/entities';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { getAppealStatsConfig } from '../appeals_analytics';

export function AppealsStats({ appeals }: { appeals: Appeal[] }) {
  const { t } = useI18n();
  const stats = getAppealStatsConfig(appeals);
  return (
    <div className="grid grid-cols-4 gap-4">
      {stats.map((s, i) => (
        <div
          key={i}
          className="flex items-center gap-4 p-4 bg-white border border-black/[0.05] rounded-xl shadow-sm"
        >
          <div className={`p-2 rounded-lg bg-black/[0.02] text-black/40`}>{s.icon}</div>
          <div>
            <div className="text-[10px] font-black uppercase tracking-widest text-black/30">
              {t(s.labelKey)}
            </div>
            <div className="text-[length:var(--text-xl)] font-bold text-black/80">{s.value}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
