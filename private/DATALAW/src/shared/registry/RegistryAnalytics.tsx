import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useI18n } from '../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../i18n/translations';
import type { AnalyticsConfig, BaseEntity, StatCard as StatCardType } from './types';

const CARD_TONES: Record<NonNullable<StatCardType['color']>, string> = {
  blue: 'bg-sky-50 text-sky-700 border-sky-200',
  green: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  purple: 'bg-fuchsia-50 text-fuchsia-700 border-fuchsia-200',
  amber: 'bg-amber-50 text-amber-700 border-amber-200',
  red: 'bg-rose-50 text-rose-700 border-rose-200',
  default: 'bg-slate-50 text-slate-700 border-slate-200',
};

const CHART_COLORS = ['#0f766e', '#2563eb', '#d97706', '#7c3aed', '#dc2626', '#475569'];

interface RegistryAnalyticsProps<T extends BaseEntity> {
  items: T[];
  config: AnalyticsConfig<T>;
}

export default function RegistryAnalytics<T extends BaseEntity>({
  items,
  config,
}: RegistryAnalyticsProps<T>) {
  const { t } = useI18n();
  const stats = config.stats(items);
  const chartData = config.chartData?.(items) ?? [];

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {stats.map((stat, index) => (
          <div
            key={`${stat.label}-${index}`}
            className={`rounded-2xl border p-4 ${CARD_TONES[stat.color ?? 'default']}`}
          >
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-white/80">
              {stat.icon}
            </div>
            <div className="text-[length:var(--text-hero)] font-black">{stat.value}</div>
            <div className="mt-1 text-[length:var(--text-sm)] font-medium opacity-80">
              {t(stat.label as TranslationKey)}
            </div>
          </div>
        ))}
      </div>

      {chartData.length > 0 && (
        <div className="rounded-3xl border border-black/5 bg-white p-5 shadow-sm">
          <div className="mb-4 text-[length:var(--text-sm)] font-semibold text-slate-700">
            {config.chartLabel ?? t('registry.analytics')}
          </div>
          <div className="h-72 min-w-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="name"
                  stroke="#64748b"
                  fontSize={12}
                  tickFormatter={(val) => t(val)}
                />
                <YAxis allowDecimals={false} stroke="#64748b" fontSize={12} />
                <Tooltip formatter={(value, name) => [value, t(name as TranslationKey)]} labelFormatter={(label) => t(label as TranslationKey)} />
                <Bar dataKey="value" radius={[10, 10, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell
                      key={`${entry.name}-${index}`}
                      fill={CHART_COLORS[index % CHART_COLORS.length]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
