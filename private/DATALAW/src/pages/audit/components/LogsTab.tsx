import { useMemo, useState } from 'react';
import { Clock, Download } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../i18n/translations';
import type { AuditEntryRecord } from '../../../services/audit/AuditService';
import type { AppLogRecord } from '../../../services/logger/LogHistoryService';
import { SearchInput } from '../../../components/ui/form/SearchInput';
import { Button } from '../../../components/ui/form/Button';
import { SegmentedControl } from '../../../components/ui/layout/SegmentedControl';

const ACTION_BADGE: Record<string, { bg: string; text: string }> = {
  LOGIN: { bg: 'bg-green-50', text: 'text-green-700' },
  LOGOUT: { bg: 'bg-slate-50', text: 'text-slate-500' },
  CREATE: { bg: 'bg-blue-50', text: 'text-blue-700' },
  UPDATE: { bg: 'bg-amber-50', text: 'text-amber-700' },
  DELETE: { bg: 'bg-red-50', text: 'text-red-700' },
  UPLOAD: { bg: 'bg-indigo-50', text: 'text-indigo-700' },
};

function dateLocale(language: string): string {
  if (language === 'sr') return 'sr-Latn-RS';
  if (language === 'en') return 'en-US';
  return 'ru-RU';
}

function formatTs(raw: string, locale: string) {
  try {
    const date = new Date(raw);
    if (isNaN(date.getTime())) return raw;
    return date.toLocaleString(locale, {
      day: '2-digit',
      month: '2-digit',
      year: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return raw;
  }
}

function ActionBadge({ action }: { action: string }) {
  const { t } = useI18n();
  const key = action.split('_')[0].toUpperCase();
  const cfg = ACTION_BADGE[key] ?? { bg: 'bg-gray-50', text: 'text-gray-500' };
  const tk = `audit.logs.action.${key}` as TranslationKey;
  const translated = t(tk);
  const label = translated === tk ? action : translated;

  return (
    <span
      className={`inline-block rounded px-2 py-0.5 text-[length:var(--text-xs)] font-bold uppercase tracking-wider ${cfg.bg} ${cfg.text}`}
    >
      {label}
    </span>
  );
}

function downloadJson(filename: string, payload: unknown) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: 'application/json;charset=utf-8',
  });
  const href = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = href;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(href);
}

export function LogsTab({
  logs,
  appLogs,
}: {
  logs: AuditEntryRecord[];
  appLogs: AppLogRecord[];
}) {
  const { t, language } = useI18n();
  const loc = dateLocale(language);
  const [filter, setFilter] = useState('');
  const [scope, setScope] = useState<'audit' | 'app'>('audit');

  const visibleLogs = scope === 'audit' ? logs : appLogs;

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return visibleLogs;

    if (scope === 'audit') {
      return logs.filter(
        (log) =>
          log.action.toLowerCase().includes(q) ||
          (log.actor_login ?? '').toLowerCase().includes(q) ||
          (log.entity_type ?? '').toLowerCase().includes(q) ||
          (log.details ?? '').toLowerCase().includes(q),
      );
    }

    return appLogs.filter(
      (log) =>
        log.level.toLowerCase().includes(q) ||
        log.tag.toLowerCase().includes(q) ||
        log.message.toLowerCase().includes(q) ||
        (log.details ?? '').toLowerCase().includes(q),
    );
  }, [appLogs, filter, logs, scope, visibleLogs]);

  if (visibleLogs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-24 text-black/20">
        <Clock size={40} strokeWidth={1} />
        <span className="text-[length:var(--text-xs)] font-semibold uppercase tracking-wider">
          {t('audit.logs.empty')}
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <div className="min-w-[260px] flex-1">
          <SearchInput
            value={filter}
            onChange={setFilter}
            placeholder={t('audit.logs.search')}
            className="h-9"
          />
        </div>
        <SegmentedControl
          active={scope}
          onChange={(id) => setScope(id as 'audit' | 'app')}
          segments={[
            { id: 'audit', label: 'Audit' },
            { id: 'app', label: 'App logs' },
          ]}
        />
        <Button
          variant="secondary"
          size="sm"
          icon={<Download size={14} />}
          onClick={() =>
            downloadJson(scope === 'audit' ? 'audit-log.json' : 'app-log.json', filtered)
          }
        >
          Export
        </Button>
        <span className="shrink-0 text-[length:var(--text-xs)] font-mono text-black/30">
          {filtered.length} / {visibleLogs.length}
        </span>
      </div>

      <div className="overflow-auto rounded-xl border border-black/[0.08] bg-white shadow-sm">
        <table className="w-full text-left text-[length:var(--text-xs)]">
          <thead className="border-b border-black/[0.05] bg-black/[0.02]">
            <tr>
              {[
                t('audit.logs.time'),
                scope === 'audit' ? t('audit.logs.user') : 'Level / tag',
                scope === 'audit' ? t('audit.logs.action') : 'Message',
                scope === 'audit' ? t('audit.logs.entity') : 'Details',
                t('audit.logs.details'),
              ].map((header) => (
                <th
                  key={header}
                  className="whitespace-nowrap px-4 py-3 text-[length:var(--text-xs)] font-bold uppercase tracking-wider text-black/35"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {scope === 'audit'
              ? (filtered as AuditEntryRecord[]).map((log) => (
                  <tr
                    key={`audit-${log.id}`}
                    className="border-b border-black/[0.03] transition-colors last:border-b-0 hover:bg-black/[0.015]"
                  >
                    <td className="whitespace-nowrap px-4 py-2.5 font-mono text-[length:var(--text-xs)] text-black/40">
                      {formatTs(log.created_at, loc)}
                    </td>
                    <td className="whitespace-nowrap px-4 py-2.5 font-medium text-black/70">
                      {log.actor_login ?? (
                        <span className="italic text-black/25">{t('audit.logs.actorSystem')}</span>
                      )}
                      {log.actor_role && (
                        <span className="ml-1.5 text-[length:var(--text-xs)] font-normal text-black/25">
                          ({log.actor_role})
                        </span>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-4 py-2.5">
                      <ActionBadge action={log.action} />
                    </td>
                    <td className="px-4 py-2.5 text-black/55">
                      <span className="font-medium">{log.entity_type}</span>
                      {log.entity_label && <span className="text-black/35">: {log.entity_label}</span>}
                    </td>
                    <td className="max-w-xs truncate px-4 py-2.5 text-black/40">
                      {log.details ?? '—'}
                    </td>
                  </tr>
                ))
              : (filtered as AppLogRecord[]).map((log) => (
                  <tr
                    key={`app-${log.id}`}
                    className="border-b border-black/[0.03] transition-colors last:border-b-0 hover:bg-black/[0.015]"
                  >
                    <td className="whitespace-nowrap px-4 py-2.5 font-mono text-[length:var(--text-xs)] text-black/40">
                      {formatTs(log.created_at, loc)}
                    </td>
                    <td className="px-4 py-2.5 text-black/70">
                      <span className="font-semibold">{log.level}</span>
                      <span className="ml-2 text-black/35">{log.tag}</span>
                    </td>
                    <td className="px-4 py-2.5 text-black/60">{log.message}</td>
                    <td className="px-4 py-2.5 text-black/45">{log.details ?? '—'}</td>
                    <td className="px-4 py-2.5 text-black/35">{log.id}</td>
                  </tr>
                ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
