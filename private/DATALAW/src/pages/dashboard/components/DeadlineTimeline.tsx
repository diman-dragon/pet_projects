// src/pages/dashboard/components/DeadlineTimeline.tsx

import { AlertTriangle, Calendar, CheckCircle2, Clock, Flag } from 'lucide-react';
import { Badge } from '../../../components/ui';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';
import type { DeadlineItem } from '../types';

interface DeadlineTimelineProps {
  items: DeadlineItem[];
}

function daysUntil(dateStr: string): number {
  const date = new Date(dateStr);
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  return Math.ceil((date.getTime() - now.getTime()) / 86400000);
}

function NodeIcon({ type, days }: { type: string; days: number }) {
  if (days < 0) return <CheckCircle2 size={13} className="text-black/20" />;
  if (type === 'court') return <Flag size={13} className="text-red-500" />;
  if (days <= 3) return <AlertTriangle size={13} className="text-orange-500" />;
  return <Clock size={13} className="text-black/35" />;
}

function nodeColor(type: string, days: number): string {
  if (days < 0) return 'border-black/[0.08] bg-black/[0.02]';
  if (type === 'court') return 'border-red-200 bg-red-50';
  if (days <= 3) return 'border-orange-200 bg-orange-50';
  if (type === 'contract') return 'border-amber-200 bg-amber-50';
  return 'border-black/[0.10] bg-white';
}

function labelColor(days: number): string {
  if (days < 0) return 'text-black/30';
  if (days === 0) return 'text-blue-600 font-bold';
  if (days <= 3) return 'text-orange-600 font-semibold';
  return 'text-black/40';
}

function dayLabel(
  days: number,
  t: (key: TranslationKey, params?: Record<string, string | number>) => string,
): string {
  if (days < 0) return t('dashboard.daysAgo', { days: Math.abs(days) });
  if (days === 0) return t('dashboard.today');
  if (days === 1) return t('dashboard.tomorrow');
  return t('dashboard.inDays', { days });
}

export function DeadlineTimeline({ items }: DeadlineTimelineProps) {
  const { t } = useI18n();
  const upcoming = items.filter((item) => daysUntil(item.date) >= 0);
  const past = items.filter((item) => daysUntil(item.date) < 0);

  return (
    <div className="flex h-full flex-col rounded-xl border border-black/[0.08] bg-white p-5 shadow-sm">
      <div className="mb-5 flex items-center gap-2">
        <div className="flex size-6 items-center justify-center rounded-lg bg-amber-50">
          <Clock size={12} className="text-amber-600" />
        </div>
        <span className="text-[10px] font-bold uppercase tracking-wider text-black/40">
          {t('dashboard.eventTimeline')}
        </span>
        {upcoming.length > 0 && (
          <Badge label={t('dashboard.activeCount', { count: upcoming.length })} variant="red" />
        )}
      </div>

      <div className="custom-scrollbar flex-1 overflow-y-auto pr-1">
        {items.length === 0 ? (
          <div className="flex min-h-[160px] flex-1 items-center justify-center">
            <div className="text-center">
              <Calendar size={28} className="mx-auto mb-2 text-black/10" />
              <p className="text-[11px] text-black/30">{t('dashboard.noUpcoming')}</p>
            </div>
          </div>
        ) : (
          <div>
            {upcoming.map((item, i) => {
              const days = daysUntil(item.date);
              const isLast = i === upcoming.length - 1 && past.length === 0;
              return (
                <div key={item.id} className="flex gap-3">
                  <div className="flex shrink-0 flex-col items-center">
                    <div
                      className={`z-10 flex h-6 w-6 items-center justify-center rounded-full border ${nodeColor(item.type, days)}`}
                    >
                      <NodeIcon type={item.type} days={days} />
                    </div>
                    {!isLast && <div className="mb-1 mt-1 w-px flex-1 bg-black/[0.07]" />}
                  </div>
                  <div
                    className={`mb-1 min-w-0 flex-1 rounded-lg border px-3 py-2 pb-3 ${nodeColor(item.type, days)}`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="truncate text-[12px] font-semibold text-black/75">
                        {item.title}
                      </p>
                      <div className="flex shrink-0 flex-col items-end">
                        <span className="whitespace-nowrap text-[9px] font-mono text-black/35">
                          {item.date}
                        </span>
                        <span className={`whitespace-nowrap text-[9px] ${labelColor(days)}`}>
                          {dayLabel(days, t)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}

            {past.length > 0 && (
              <>
                {upcoming.length > 0 && (
                  <div className="my-3 flex items-center gap-2">
                    <div className="h-px flex-1 bg-black/[0.06]" />
                    <span className="text-[8px] font-black uppercase tracking-widest text-black/20">
                      {t('dashboard.past')}
                    </span>
                    <div className="h-px flex-1 bg-black/[0.06]" />
                  </div>
                )}
                {past.slice(0, 3).map((item, i, arr) => {
                  const days = daysUntil(item.date);
                  return (
                    <div key={item.id} className="flex gap-3">
                      <div className="flex shrink-0 flex-col items-center">
                        <div className="z-10 flex h-6 w-6 items-center justify-center rounded-full border border-black/[0.08] bg-black/[0.02]">
                          <CheckCircle2 size={12} className="text-black/15" />
                        </div>
                        {i < arr.length - 1 && (
                          <div className="mb-1 mt-1 w-px flex-1 bg-black/[0.05]" />
                        )}
                      </div>
                      <div className="mb-1 min-w-0 flex-1 rounded-lg border border-black/[0.05] bg-black/[0.01] px-3 py-2 pb-3">
                        <div className="flex items-center justify-between gap-2">
                          <p className="truncate text-[11px] font-medium text-black/30">
                            {item.title}
                          </p>
                          <span className="whitespace-nowrap text-[9px] font-mono text-black/20">
                            {dayLabel(days, t)}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
