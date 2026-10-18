// src/pages/dashboard/components/ClientDistribution.tsx

import { Users } from 'lucide-react';
import { Pie, PieChart, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { useI18n } from '../../../i18n/AppLanguageProvider';

const CHART_COLORS = ['#0067c0', '#e5e5e5', '#10b981', '#f59e0b'];

interface ClientDistributionProps {
  data: { name: string; value: number; color: string }[];
}

export function ClientDistribution({ data }: ClientDistributionProps) {
  const { t } = useI18n();
  const hasData = data && data.length > 0;
  const total = hasData ? data.reduce((sum, item) => sum + item.value, 0) : 0;

  return (
    <div className="flex h-full flex-col rounded-xl border border-black/[0.08] bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center gap-2">
        <div className="flex size-6 items-center justify-center rounded-lg bg-blue-50">
          <Users size={12} className="text-blue-600" />
        </div>
        <span className="text-[10px] font-bold uppercase tracking-wider text-black/40">
          {t('dashboard.clientBase')}
        </span>
      </div>

      <div className="flex flex-1 flex-col">
        {hasData ? (
          <>
            <div className="h-[180px] min-w-0">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={data}
                    innerRadius={50}
                    outerRadius={70}
                    paddingAngle={4}
                    dataKey="value"
                    stroke="none"
                  >
                    {data.map((_, idx) => (
                      <Cell key={idx} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      borderRadius: '8px',
                      border: '1px solid rgba(0,0,0,0.05)',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                      fontSize: '11px',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3 border-t border-black/[0.05] pt-3">
              {data.map((item, idx) => {
                const percent = total > 0 ? Math.round((item.value / total) * 100) : 0;
                return (
                  <div key={idx} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div
                        className="size-2 rounded-full"
                        style={{ backgroundColor: CHART_COLORS[idx % CHART_COLORS.length] }}
                      />
                      <span className="text-[11px] font-medium text-black/60">{item.name}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-[13px] font-bold text-black/80">{item.value}</span>
                      <span className="text-[9px] text-black/30">({percent}%)</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        ) : (
          <div className="flex flex-1 items-center justify-center">
            <div className="text-center">
              <Users size={32} className="mx-auto mb-2 text-black/10" />
              <p className="text-[11px] text-black/30">{t('audit.analytics.noData')}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
