// src/pages/dashboard/components/LogPanel.tsx

import { Terminal, TrendingUp } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { LogItem } from '../types';

interface LogPanelProps {
  logs: LogItem[];
}

export function LogPanel({ logs }: LogPanelProps) {
  const { t } = useI18n();

  if (!logs || logs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Terminal size={32} className="mb-2 text-black/10" />
        <p className="text-[11px] text-black/30">{t('dashboard.systemLogsEmpty')}</p>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {logs.map((log) => (
        <div
          key={log.id}
          className="group flex items-start gap-3 rounded-lg p-2.5 transition-all hover:bg-black/[0.02]"
        >
          <div className="flex size-6 shrink-0 items-center justify-center rounded bg-black/[0.03] transition-colors group-hover:bg-[var(--app-accent)] group-hover:text-white">
            <Terminal size={10} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-medium text-black/80">{log.msg}</span>
            </div>
            <div className="mt-1 flex items-center gap-2">
              <span className="rounded bg-black/[0.03] px-1.5 py-0.5 text-[8px] font-bold uppercase text-black/30">
                {log.user}
              </span>
              <span className="text-[8px] font-mono text-black/20">{log.time}</span>
            </div>
          </div>
          <TrendingUp
            size={10}
            className="text-black/10 transition-colors group-hover:text-black/30"
          />
        </div>
      ))}
    </div>
  );
}
