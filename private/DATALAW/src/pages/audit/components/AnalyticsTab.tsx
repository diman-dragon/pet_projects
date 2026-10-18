// src/pages/audit/components/AnalyticsTab.tsx

import { useMemo } from 'react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { CaseTypeStat, DbInfo, MonthStat, UserRecord } from '../types';

interface AnalyticsTabProps {
  analytics: {
    clientsByMonth: MonthStat[];
    casesByType: CaseTypeStat[];
  };
  databases: DbInfo[];
  users: UserRecord[];
}

function BeautifulBarChart({
  data,
  color,
}: {
  data: { label: string; value: number }[];
  color: string;
}) {
  const max = Math.max(...data.map((d) => d.value), 1);

  return (
    <div className="relative h-64 pt-6">
      <div className="absolute inset-0 flex items-end gap-2 px-2">
        {data.map((item, i) => {
          const height = Math.max((item.value / max) * 100, 8);
          return (
            <div key={i} className="group relative flex flex-1 flex-col items-center justify-end">
              <div className="mb-1.5 text-[length:var(--text-xs)] font-mono text-black/40 opacity-0 transition-opacity group-hover:opacity-100">
                {item.value.toLocaleString('ru-RU')}
              </div>
              <div
                className="w-full rounded-t-xl transition-all duration-300 hover:brightness-110"
                style={{
                  height: `${height}%`,
                  background: `linear-gradient(to top, ${color}88, ${color})`,
                  boxShadow: '0 4px 12px -2px rgba(0,0,0,0.1)',
                }}
              />
              <div className="mt-2 w-full truncate text-center text-[length:var(--text-xs)] leading-tight text-black/30">
                {item.label}
              </div>
            </div>
          );
        })}
      </div>
      <div className="absolute bottom-0 left-0 right-0 h-px bg-black/[0.08]" />
    </div>
  );
}

function BeautifulPieChart({
  data,
  emptyLabel,
}: {
  data: { label: string; value: number }[];
  emptyLabel: string;
}) {
  const total = data.reduce((sum, item) => sum + item.value, 0);
  let startAngle = 0;

  if (total === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-[length:var(--text-sm)] italic text-black/25">
        {emptyLabel}
      </div>
    );
  }

  return (
    <div className="flex justify-center">
      <div className="relative h-64 w-64">
        <svg viewBox="0 0 200 200" className="h-full w-full -rotate-12">
          {data.map((item, i) => {
            const percentage = item.value / total;
            const angle = percentage * 360;
            const endAngle = startAngle + angle;

            const x1 = 100 + 80 * Math.cos((startAngle - 90) * (Math.PI / 180));
            const y1 = 100 + 80 * Math.sin((startAngle - 90) * (Math.PI / 180));
            const x2 = 100 + 80 * Math.cos((endAngle - 90) * (Math.PI / 180));
            const y2 = 100 + 80 * Math.sin((endAngle - 90) * (Math.PI / 180));

            const largeArc = angle > 180 ? 1 : 0;
            const path = `M 100 100 L ${x1} ${y1} A 80 80 0 ${largeArc} 1 ${x2} ${y2} Z`;
            const segmentColor = ['#0067c0', '#7c3aed', '#059669', '#d97706', '#dc2626', '#0891b2'][
              i % 6
            ];

            startAngle = endAngle;

            return (
              <path
                key={i}
                d={path}
                fill={segmentColor}
                stroke="#fff"
                strokeWidth="3"
                className="transition-all hover:brightness-110"
              >
                <title>{`${item.label}: ${item.value}`}</title>
              </path>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

export function AnalyticsTab({ analytics, databases, users }: AnalyticsTabProps) {
  const { t } = useI18n();

  const totalClients = useMemo(() => {
    const db = databases.find((item) => item.key === 'clients_main');
    return db?.tables.find((table) => table.name === 'clients')?.rowCount ?? 0;
  }, [databases]);

  const totalCases = useMemo(() => {
    const db = databases.find((item) => item.key === 'clients_main');
    return db?.tables.find((table) => table.name === 'cases')?.rowCount ?? 0;
  }, [databases]);

  const roleData = useMemo(() => {
    const counts: Record<string, number> = {};
    users.forEach((user) => {
      counts[user.role] = (counts[user.role] || 0) + 1;
    });
    return Object.entries(counts).map(([label, value]) => ({ label, value }));
  }, [users]);

  const monthData = analytics.clientsByMonth.map((item) => ({
    label: item.month.slice(5),
    value: item.count,
  }));

  const typeData = analytics.casesByType.map((item) => ({
    label: item.case_type ?? t('audit.analytics.unknownCaseType'),
    value: item.count,
  }));

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {[
          {
            label: t('audit.analytics.totalClients'),
            value: totalClients,
            color: 'var(--app-accent)',
          },
          { label: t('audit.analytics.totalCases'), value: totalCases, color: '#7c3aed' },
          { label: t('audit.analytics.totalUsers'), value: users.length, color: '#059669' },
          {
            label: t('audit.analytics.totalMonths'),
            value: analytics.clientsByMonth.length,
            color: '#d97706',
          },
        ].map((stat, i) => (
          <div key={i} className="rounded-2xl border border-black/[0.06] bg-white p-6 shadow-sm">
            <p className="text-[length:var(--text-xs)] font-bold uppercase tracking-[0.5px] text-black/40">
              {stat.label}
            </p>
            <p className="mt-2 text-[length:var(--text-display)] font-black" style={{ color: stat.color }}>
              {stat.value.toLocaleString('ru-RU')}
            </p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="rounded-2xl border border-black/[0.06] bg-white p-6 shadow-sm">
          <h4 className="mb-5 text-[length:var(--text-xs)] font-bold uppercase tracking-wider text-black/40">
            {t('audit.analytics.growth')}
          </h4>
          {monthData.length > 0 ? (
            <BeautifulBarChart data={monthData} color="#0067c0" />
          ) : (
            <div className="flex h-64 items-center justify-center text-[length:var(--text-sm)] text-black/25">
              {t('audit.analytics.noData')}
            </div>
          )}
        </div>

        <div className="rounded-2xl border border-black/[0.06] bg-white p-6 shadow-sm">
          <h4 className="mb-5 text-[length:var(--text-xs)] font-bold uppercase tracking-wider text-black/40">
            {t('audit.analytics.caseTypes')}
          </h4>
          <BeautifulPieChart data={typeData} emptyLabel={t('audit.analytics.noChartData')} />
        </div>

        <div className="rounded-2xl border border-black/[0.06] bg-white p-6 shadow-sm">
          <h4 className="mb-5 text-[length:var(--text-xs)] font-bold uppercase tracking-wider text-black/40">
            {t('audit.analytics.roles')}
          </h4>
          <BeautifulPieChart data={roleData} emptyLabel={t('audit.analytics.noChartData')} />
        </div>
      </div>
    </div>
  );
}
