import { useCallback, useEffect, useMemo, useState } from 'react';
import { Database, RefreshCw, ShieldAlert, Trash2 } from 'lucide-react';
import { Button } from '../../../components/ui/form/Button';
import { SearchInput } from '../../../components/ui/form/SearchInput';
import { SegmentedControl } from '../../../components/ui/layout/SegmentedControl';
import { UiPanel } from '../../../components/ui/atoms/UiPanel';
import { UiText } from '../../../components/ui/atoms/UiText';
import type { AppLogRecord } from '../../../services/logger/LogHistoryService';
import { LogHistoryService } from '../../../services/logger/LogHistoryService';
import { createLogger } from '../../../services/logger/Logger';

const log = createLogger('SettingsLogsSection');

type LogLevelFilter = 'ALL' | 'CRITICAL' | 'ERROR' | 'WARN' | 'INFO' | 'DEBUG';

function formatTimestamp(raw: string): string {
  try {
    const date = new Date(raw);
    if (Number.isNaN(date.getTime())) return raw;
    return date.toLocaleString();
  } catch {
    return raw;
  }
}

const LEVEL_COLORS: Record<string, string> = {
  CRITICAL: 'bg-red-100 text-red-700',
  ERROR: 'bg-rose-100 text-rose-700',
  WARN: 'bg-amber-100 text-amber-700',
  INFO: 'bg-sky-100 text-sky-700',
  DEBUG: 'bg-slate-100 text-slate-600',
};

export function SettingsLogsSection() {
  const [loading, setLoading] = useState(true);
  const [logs, setLogs] = useState<AppLogRecord[]>([]);
  const [search, setSearch] = useState('');
  const [level, setLevel] = useState<LogLevelFilter>('ALL');
  const [tag, setTag] = useState('ALL');
  const [stats, setStats] = useState<{
    total: number;
    critical: number;
    error: number;
    warn: number;
    info: number;
    debug: number;
    tags: string[];
  }>({
    total: 0,
    critical: 0,
    error: 0,
    warn: 0,
    info: 0,
    debug: 0,
    tags: [],
  });

  const loadLogs = useCallback(async () => {
    setLoading(true);
    try {
      const [rows, nextStats] = await Promise.all([
        LogHistoryService.queryLogs({
          level,
          tag: tag === 'ALL' ? undefined : tag,
          search,
          limit: 500,
        }),
        LogHistoryService.getLogStats(),
      ]);
      setLogs(rows);
      setStats(nextStats);
    } catch (error) {
      log.error('Failed to load app logs for settings', error);
    } finally {
      setLoading(false);
    }
  }, [level, search, tag]);

  useEffect(() => {
    void loadLogs();
  }, [loadLogs]);

  const levelSegments = useMemo(
    () => [
      { id: 'ALL', label: `Все (${stats.total})` },
      { id: 'CRITICAL', label: `Critical (${stats.critical})` },
      { id: 'ERROR', label: `Error (${stats.error})` },
      { id: 'WARN', label: `Warn (${stats.warn})` },
      { id: 'INFO', label: `Info (${stats.info})` },
      { id: 'DEBUG', label: `Debug (${stats.debug})` },
    ],
    [stats],
  );

  const tagSegments = useMemo(
    () => [
      { id: 'ALL', label: 'Все теги' },
      ...stats.tags.slice(0, 8).map((item) => ({
        id: item,
        label: item.length > 18 ? `${item.slice(0, 16)}...` : item,
      })),
    ],
    [stats.tags],
  );

  return (
    <div className="flex flex-col gap-6">
      <UiPanel padding="lg" mica elevation="md" className="rounded-[24px]">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="max-w-3xl">
            <UiText as="p" size="xs" weight="bold" tone="muted" className="label-caps">
              System log registry
            </UiText>
            <UiText as="h3" size="lg" weight="semibold" heading className="mt-3">
              Журнал приложения
            </UiText>
            <UiText as="p" size="sm" tone="secondary" className="mt-2">
              Здесь хранится всё, что раньше засоряло консоль: предупреждения, обычные ошибки,
              служебные сообщения и детали выполнения. В консоль должны попадать только критические ошибки.
            </UiText>
          </div>

          <div className="grid min-w-[240px] grid-cols-2 gap-3">
            <UiPanel padding="md" mica elevation="sm" className="rounded-[18px]">
              <UiText as="p" size="xs" tone="muted">
                Всего записей
              </UiText>
              <UiText as="p" size="xl" weight="semibold" className="mt-2">
                {stats.total}
              </UiText>
            </UiPanel>
            <UiPanel padding="md" mica elevation="sm" className="rounded-[18px]">
              <UiText as="p" size="xs" tone="muted">
                Критические
              </UiText>
              <UiText as="p" size="xl" weight="semibold" className="mt-2 text-red-600">
                {stats.critical}
              </UiText>
            </UiPanel>
          </div>
        </div>
      </UiPanel>

      <UiPanel padding="lg" mica elevation="md" className="rounded-[24px]">
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="min-w-[280px] flex-1">
              <SearchInput
                value={search}
                onChange={setSearch}
                placeholder="Поиск по сообщению, деталям или тегу"
                className="h-9"
              />
            </div>
            <Button variant="secondary" size="sm" icon={<RefreshCw size={14} />} onClick={() => void loadLogs()}>
              Обновить
            </Button>
            <Button
              variant="outline"
              size="sm"
              icon={<Trash2 size={14} />}
              onClick={async () => {
                await LogHistoryService.clearLogs();
                await loadLogs();
              }}
            >
              Очистить
            </Button>
          </div>

          <div className="flex flex-col gap-3">
            <SegmentedControl
              active={level}
              onChange={(id) => setLevel(id as LogLevelFilter)}
              segments={levelSegments}
            />
            {stats.tags.length > 0 ? (
              <SegmentedControl active={tag} onChange={setTag} segments={tagSegments} />
            ) : null}
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <UiPanel padding="md" mica elevation="sm" className="rounded-[18px]">
              <div className="flex items-center gap-3">
                <Database size={16} className="text-[var(--app-accent)]" />
                <div>
                  <UiText as="p" size="xs" tone="muted">
                    Хранение
                  </UiText>
                  <UiText as="p" size="sm" weight="medium" className="mt-1">
                    SQLite `app_logs`
                  </UiText>
                </div>
              </div>
            </UiPanel>
            <UiPanel padding="md" mica elevation="sm" className="rounded-[18px]">
              <div className="flex items-center gap-3">
                <ShieldAlert size={16} className="text-red-500" />
                <div>
                  <UiText as="p" size="xs" tone="muted">
                    Консоль
                  </UiText>
                  <UiText as="p" size="sm" weight="medium" className="mt-1">
                    Только critical
                  </UiText>
                </div>
              </div>
            </UiPanel>
            <UiPanel padding="md" mica elevation="sm" className="rounded-[18px]">
              <UiText as="p" size="xs" tone="muted">
                Показано
              </UiText>
              <UiText as="p" size="sm" weight="medium" className="mt-1">
                {logs.length} записей
              </UiText>
            </UiPanel>
          </div>

          <div className="overflow-auto rounded-[20px] border border-[var(--app-border)] bg-[var(--app-surface)]">
            <table className="w-full min-w-[980px] text-left text-sm">
              <thead className="border-b border-[var(--app-border)] bg-black/[0.02]">
                <tr>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">
                    Время
                  </th>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">
                    Уровень
                  </th>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">
                    Тег
                  </th>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">
                    Сообщение
                  </th>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">
                    Детали
                  </th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={5} className="px-4 py-12 text-center text-sm text-[var(--app-text-muted)]">
                      Загрузка журнала...
                    </td>
                  </tr>
                ) : logs.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-4 py-12 text-center text-sm text-[var(--app-text-muted)]">
                      По текущему фильтру записей нет.
                    </td>
                  </tr>
                ) : (
                  logs.map((entry) => (
                    <tr
                      key={entry.id}
                      className="border-b border-[var(--app-border)]/70 align-top transition-colors last:border-b-0 hover:bg-black/[0.02]"
                    >
                      <td className="whitespace-nowrap px-4 py-3 text-xs text-[var(--app-text-muted)]">
                        {formatTimestamp(entry.created_at)}
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={[
                            'inline-flex rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-wider',
                            LEVEL_COLORS[entry.level] ?? 'bg-slate-100 text-slate-700',
                          ].join(' ')}
                        >
                          {entry.level}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-[var(--app-text-secondary)]">
                        {entry.tag}
                      </td>
                      <td className="max-w-[320px] px-4 py-3 text-sm text-[var(--app-text)]">
                        {entry.message}
                      </td>
                      <td className="max-w-[420px] whitespace-pre-wrap px-4 py-3 text-xs text-[var(--app-text-secondary)]">
                        {entry.details || '—'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </UiPanel>
    </div>
  );
}
