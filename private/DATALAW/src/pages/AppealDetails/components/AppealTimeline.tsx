// src/pages/AppealDetails/components/AppealTimeline.tsx

import { Clock, AlertTriangle, CheckCircle2, Calendar, FileText, Flag } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { appLanguageToBcp47 } from '../../../i18n/config';
import type { TranslationKey } from '../../../i18n/translations';
import type { DeadlineEvent } from '../types';
import { formatDateShort } from '../config';

interface AppealTimelineProps {
  events: DeadlineEvent[];
  onOpenFile?: (path: string) => void;
}

type TFn = (key: TranslationKey, params?: Record<string, string | number>) => string;

function daysLabel(
  days: number,
  isToday: boolean,
  isPast: boolean,
  t: TFn,
): { text: string; cls: string } {
  if (isPast)
    return { text: t('appeal.timeline.daysAgo', { days: Math.abs(days) }), cls: 'text-black/25' };
  if (isToday) return { text: t('appeal.timeline.today'), cls: 'text-blue-600 font-bold' };
  if (days === 1)
    return { text: t('appeal.timeline.tomorrow'), cls: 'text-orange-600 font-semibold' };
  return { text: t('appeal.timeline.inDays', { days }), cls: 'text-black/40' };
}

function eventNow(d: Date) {
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  return Math.ceil((d.getTime() - now.getTime()) / 86400000);
}

function EventIcon({ event }: { event: DeadlineEvent }) {
  if (event.kind === 'appeal_date') return <Flag size={14} className="text-blue-500" />;
  if (event.kind === 'response_due')
    return (
      <Clock
        size={14}
        className={event.isUrgent && !event.isPast ? 'text-orange-500' : 'text-black/30'}
      />
    );
  if (event.isPast) return <CheckCircle2 size={14} className="text-black/20" />;
  if (event.isUrgent) return <AlertTriangle size={14} className="text-orange-500" />;
  return <FileText size={14} className="text-black/35" />;
}

function TimelineNode({
  event,
  isLast,
  onOpenFile,
  localeTag,
  t,
}: {
  event: DeadlineEvent;
  isLast: boolean;
  onOpenFile?: (p: string) => void;
  localeTag: string;
  t: TFn;
}) {
  const days = eventNow(event.date);
  const label = daysLabel(days, event.isToday, event.isPast, t);
  const clickable = !!event.filePath && !!onOpenFile;
  const eventLabel = t(event.labelKey as TranslationKey);

  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center shrink-0">
        <div
          className={`w-7 h-7 rounded-full flex items-center justify-center border-2 z-10
                    ${
                      event.isPast
                        ? 'border-black/10 bg-black/[0.03]'
                        : event.isUrgent
                          ? 'border-orange-300 bg-orange-50'
                          : event.isToday
                            ? 'border-blue-400 bg-blue-50'
                            : 'border-black/15 bg-white'
                    }
                `}
        >
          <EventIcon event={event} />
        </div>
        {!isLast && <div className="w-px flex-1 mt-1 mb-1 bg-black/[0.07]" />}
      </div>

      <div
        className={`flex-1 pb-5 min-w-0 ${clickable ? 'cursor-pointer' : ''}`}
        onClick={() => clickable && onOpenFile!(event.filePath!)}
      >
        <div
          className={`rounded-xl border px-3 py-2.5 transition-all
                    ${
                      event.isPast
                        ? 'border-black/[0.06] bg-black/[0.01]'
                        : event.isUrgent
                          ? 'border-orange-200 bg-orange-50/50 hover:bg-orange-50'
                          : event.isToday
                            ? 'border-blue-200 bg-blue-50/50 hover:bg-blue-50'
                            : 'border-black/[0.07] bg-white hover:bg-black/[0.02]'
                    }
                `}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0 flex-1">
              <p
                className={`text-[length:var(--text-xs)] font-semibold leading-tight truncate
                                ${event.isPast ? 'text-black/35' : 'text-black/75'}
                            `}
              >
                {eventLabel}
              </p>
              {event.fileName && event.fileName !== eventLabel && (
                <p className="text-[length:var(--text-xs)] font-mono text-black/25 truncate mt-0.5">
                  {event.fileName}
                </p>
              )}
            </div>
            <div className="flex flex-col items-end shrink-0 gap-0.5">
              <span className="text-[length:var(--text-xs)] font-mono text-black/40 whitespace-nowrap">
                {formatDateShort(event.date, localeTag)}
              </span>
              <span className={`text-[length:var(--text-xs)] whitespace-nowrap ${label.cls}`}>
                {label.text}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function AppealTimeline({ events, onOpenFile }: AppealTimelineProps) {
  const { t, language } = useI18n();
  const localeTag = appLanguageToBcp47(language);

  if (events.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 py-16 text-center">
        <div className="w-10 h-10 rounded-full bg-black/[0.04] flex items-center justify-center">
          <Calendar size={18} className="text-black/20" />
        </div>
        <p className="text-[length:var(--text-sm)] text-black/35 font-medium">
          {t('appeal.timeline.emptyTitle')}
        </p>
        <p className="text-[length:var(--text-xs)] text-black/25">
          {t('appeal.timeline.emptyHint')}
        </p>
      </div>
    );
  }

  const upcoming = events.filter((e) => !e.isPast);
  const past = events.filter((e) => e.isPast);

  return (
    <div className="max-w-xl mx-auto py-2">
      {upcoming.length > 0 && (
        <div className="mb-2">
          <div className="text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/25 mb-4 flex items-center gap-2">
            <span>{t('appeal.timeline.upcoming')}</span>
            <span className="px-1.5 py-0.5 rounded-full bg-blue-100 text-blue-600">
              {upcoming.length}
            </span>
          </div>
          {upcoming.map((event, i) => (
            <TimelineNode
              key={event.id}
              event={event}
              isLast={i === upcoming.length - 1 && past.length === 0}
              onOpenFile={onOpenFile}
              localeTag={localeTag}
              t={t}
            />
          ))}
        </div>
      )}

      {past.length > 0 && (
        <div>
          {upcoming.length > 0 && (
            <div className="flex items-center gap-3 my-4">
              <div className="flex-1 h-px bg-black/[0.06]" />
              <span className="text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/20">
                {t('appeal.timeline.past')}
              </span>
              <div className="flex-1 h-px bg-black/[0.06]" />
            </div>
          )}
          {past.map((event, i) => (
            <TimelineNode
              key={event.id}
              event={event}
              isLast={i === past.length - 1}
              onOpenFile={onOpenFile}
              localeTag={localeTag}
              t={t}
            />
          ))}
        </div>
      )}
    </div>
  );
}
