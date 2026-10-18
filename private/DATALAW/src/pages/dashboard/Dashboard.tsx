// src/pages/dashboard/Dashboard.tsx

import { Activity, AlertCircle, ChevronRight, Database, TrendingUp } from 'lucide-react';
import Calendar from 'react-calendar';
import 'react-calendar/dist/Calendar.css';
import { Badge, Button, StatusDot } from '../../components/ui';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { PageBase } from '../../shared/ui/PageBase';
import { ClientDistribution, DeadlineTimeline, LogPanel, StatCard } from './components';
import { useDashboardLogic } from './useDashboardLogic';
import type { CalendarValue, DeadlineItem } from './types';

export default function Dashboard() {
  const { t } = useI18n();
  const { date, setDate, isReady, stats, error } = useDashboardLogic();

  if (error && error.includes('TABLE_NOT_FOUND')) {
    return (
      <PageBase
        title={t('dashboard.initialization')}
        subtitle={t('dashboard.initializationWait')}
        icon={<Database size={22} strokeWidth={1.8} />}
      >
        <div className="flex flex-col items-center justify-center gap-4 py-20">
          <div className="flex size-16 items-center justify-center rounded-2xl bg-amber-50">
            <AlertCircle size={32} className="text-amber-500" />
          </div>
          <h3 className="text-[length:var(--text-md)] font-bold leading-[var(--leading-heading)] text-[var(--app-text-secondary)]">
            {t('dashboard.dbMissing')}
          </h3>
          <p className="max-w-sm text-center text-[length:var(--text-xs)] leading-[var(--leading-body)] text-[var(--app-text-muted)]">
            {t('dashboard.dbMissingBody')}
          </p>
          <div className="mt-4">
            <div className="size-5 animate-spin rounded-full border-2 border-[var(--app-accent)] border-t-transparent" />
          </div>
        </div>
      </PageBase>
    );
  }

  return (
    <PageBase
      title={t('dashboard.title')}
      subtitle={t('dashboard.subtitle')}
      icon={<TrendingUp size={22} strokeWidth={1.8} />}
      extra={
        <div className="flex items-center gap-3">
          <div
            className={`flex items-center gap-2 rounded-full border px-3 py-1.5 transition-all duration-500 ${
              isReady
                ? 'border-emerald-200/50 bg-emerald-50/50'
                : 'border-amber-200/50 bg-amber-50/50'
            }`}
          >
            <StatusDot variant={isReady ? 'green' : 'amber'} />
            <span
              className={`text-[length:var(--text-xs)] font-bold uppercase leading-[var(--leading-heading)] tracking-wider ${
                isReady ? 'text-emerald-600' : 'text-amber-600'
              }`}
            >
              {isReady ? t('dashboard.dbReady') : t('dashboard.dbSync')}
            </span>
          </div>
        </div>
      }
    >
      <div className="space-y-6">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            title={t('dashboard.cases')}
            value={stats.docsCount}
            sub={t('dashboard.casesSub')}
            icon="briefcase"
            accent="blue"
          />
          <StatCard
            title={t('dashboard.clients')}
            value={stats.clientsCount}
            sub={t('dashboard.clientsSub')}
            icon="users"
            accent="green"
          />
          <StatCard
            title={t('dashboard.lines')}
            value={stats.winRate}
            sub={t('dashboard.linesSub')}
            icon="network"
            accent="amber"
          />
          <StatCard
            title={t('dashboard.docs')}
            value={stats.pnpProgress}
            sub={t('dashboard.docsSub')}
            icon="file-text"
            accent="violet"
          />
        </div>

        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12 lg:col-span-4">
            <ClientDistribution data={stats.clientDist} />
          </div>
          <div className="col-span-12 lg:col-span-4">
            <DeadlineTimeline items={stats.deadlines} />
          </div>
          <div className="col-span-12 lg:col-span-4">
            <CalendarTimelineCard date={date} setDate={setDate} deadlines={stats.deadlines} />
          </div>
        </div>

        <div className="overflow-hidden rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-raised)] shadow-[var(--app-shadow-sm)]">
          <div className="flex items-center justify-between border-b border-[var(--app-border)] bg-[color-mix(in_srgb,var(--app-text)_2%,transparent)] px-5 py-3">
            <div className="flex items-center gap-2">
              <Activity size={14} strokeWidth={2} className="text-[var(--app-text-muted)]" />
              <span className="text-[length:var(--text-xs)] font-bold uppercase leading-[var(--leading-heading)] tracking-wider text-[var(--app-text-muted)]">
                {t('dashboard.systemActivity')}
              </span>
              <Badge label={t('dashboard.badgeLive')} variant="red" />
            </div>
            <Button variant="ghost" size="sm" icon={<ChevronRight size={12} />}>
              {t('dashboard.allLogs')}
            </Button>
          </div>
          <div className="custom-scrollbar max-h-[300px] overflow-y-auto p-4">
            <LogPanel logs={stats.recentLogs || []} />
          </div>
        </div>
      </div>

      {error && !error.includes('TABLE_NOT_FOUND') && (
        <div className="fixed bottom-6 right-6 z-50">
          <div className="flex items-center gap-2 rounded-lg bg-[var(--app-danger)] px-3 py-2 text-white shadow-lg">
            <AlertCircle size={14} />
            <span className="text-[length:var(--text-xs)] font-bold uppercase leading-[var(--leading-heading)]">
              {error}
            </span>
          </div>
        </div>
      )}
    </PageBase>
  );
}

function CalendarTimelineCard({
  date,
  setDate,
  deadlines,
}: {
  date: CalendarValue;
  setDate: (value: CalendarValue) => void;
  deadlines: DeadlineItem[];
}) {
  const { t, language } = useI18n();
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const upcoming = deadlines.filter((item) => new Date(item.date) >= today).slice(0, 4);
  const eventDates = new Set(deadlines.map((item) => item.date));
  const locale = language === 'sr' ? 'sr-Latn-RS' : language === 'en' ? 'en-US' : 'ru-RU';

  return (
    <div className="flex h-full flex-col rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-raised)] p-5 shadow-[var(--app-shadow-sm)]">
      <div className="mb-4 flex items-center gap-2">
        <div className="flex size-6 items-center justify-center rounded-lg bg-purple-50">
          <Activity size={12} strokeWidth={2} className="text-purple-600" />
        </div>
        <span className="text-[length:var(--text-xs)] font-bold uppercase leading-[var(--leading-heading)] tracking-wider text-[var(--app-text-muted)]">
          {t('dashboard.calendarTimeline')}
        </span>
      </div>

      <div className="mb-4 flex justify-center">
        <Calendar
          onChange={(value) => setDate(value as CalendarValue)}
          value={date as CalendarValue}
          locale={locale}
          className="!border-none !shadow-none !text-[length:var(--text-xs)] !leading-[var(--leading-body)]"
          tileClassName={({ date: currentDate }) => {
            const iso = currentDate.toISOString().slice(0, 10);
            return eventDates.has(iso) ? '!bg-amber-100 !rounded-full !font-bold' : null;
          }}
        />
      </div>

      <div className="mb-3 flex items-center gap-2">
        <div className="h-px flex-1 bg-[color-mix(in_srgb,var(--app-text)_8%,transparent)]" />
        <span className="text-[length:var(--text-xs)] font-black uppercase leading-[var(--leading-heading)] tracking-widest text-[var(--app-text-muted)]">
          {t('dashboard.upcoming')}
        </span>
        <div className="h-px flex-1 bg-[color-mix(in_srgb,var(--app-text)_8%,transparent)]" />
      </div>

      <div className="flex-1 space-y-1.5 overflow-y-auto">
        {upcoming.length === 0 ? (
          <p className="py-4 text-center text-[length:var(--text-xs)] leading-[var(--leading-body)] text-[var(--app-text-muted)]">
            {t('dashboard.noUpcoming')}
          </p>
        ) : (
          upcoming.map((item) => {
            const itemDate = new Date(item.date);
            const days = Math.ceil((itemDate.getTime() - today.getTime()) / 86400000);
            const isToday = days === 0;
            const isUrgent = days <= 3;

            return (
              <div
                key={item.id}
                className={`flex items-center gap-2 rounded-lg border px-2.5 py-2 text-[length:var(--text-xs)] leading-[var(--leading-body)] ${
                  isToday
                    ? 'border-blue-200 bg-blue-50'
                    : isUrgent
                      ? 'border-orange-200 bg-orange-50'
                      : 'border-[var(--app-border)] bg-[color-mix(in_srgb,var(--app-text)_2%,transparent)]'
                }`}
              >
                <div
                  className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                    item.type === 'court'
                      ? 'bg-red-500'
                      : item.type === 'contract'
                        ? 'bg-amber-500'
                        : 'bg-blue-500'
                  }`}
                />
                <span className="flex-1 truncate font-medium text-[var(--app-text-secondary)]">
                  {item.title}
                </span>
                <span
                  className={`shrink-0 font-mono text-[length:var(--text-xs)] leading-[var(--leading-body)] ${
                    isToday
                      ? 'font-bold text-blue-600'
                      : isUrgent
                        ? 'text-orange-600'
                        : 'text-[var(--app-text-muted)]'
                  }`}
                >
                  {isToday ? t('dashboard.today') : t('dashboard.inDays', { days })}
                </span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
