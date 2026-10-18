// src/pages/audit/Audit.tsx

import { useState } from 'react';
import {
  Activity,
  ChartColumn,
  Database,
  FolderTree,
  LayoutDashboard,
  Loader2,
  ScrollText,
  Users,
} from 'lucide-react';
import { StatusDot } from '../../components/ui';
import { SegmentedControl } from '../../components/ui/layout/SegmentedControl';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { PageBase } from '../../shared/ui/PageBase';
import { AnalyticsTab, AuditOverview, DbTab, FsTab, LogsTab, UsersTab } from './components/index';
import { useAuditLogic } from './useAuditLogic';

export default function Audit({ rootPath }: { rootPath: string }) {
  const { t } = useI18n();
  const { fsTree, databases, logs, appLogs, users, analytics, configPath, loading } =
    useAuditLogic(rootPath);
  const [activeTab, setActiveTab] = useState('overview');

  const totalRows = databases.reduce(
    (acc, db) => acc + db.tables.reduce((sum, table) => sum + table.rowCount, 0),
    0,
  );

  if (loading) {
    return (
      <PageBase
        title={t('audit.loadingTitle')}
        subtitle={t('audit.loadingSubtitle')}
        icon={<Activity size={22} />}
        loading
      >
        <div className="flex flex-col items-center justify-center gap-3 py-20">
          <Loader2 className="animate-spin text-black/20" size={40} strokeWidth={1} />
          <span className="text-[length:var(--text-xs)] font-bold uppercase tracking-[0.2em] text-black/30">
            {t('audit.loading')}
          </span>
        </div>
      </PageBase>
    );
  }

  const segments = [
    { id: 'overview', label: t('audit.tab.overview'), icon: <LayoutDashboard size={14} /> },
    { id: 'analytics', label: t('audit.tab.analytics'), icon: <ChartColumn size={14} /> },
    { id: 'users', label: t('audit.tab.users'), icon: <Users size={14} /> },
    { id: 'fs', label: t('audit.tab.fs'), icon: <FolderTree size={14} /> },
    { id: 'db', label: t('audit.tab.db'), icon: <Database size={14} /> },
    { id: 'logs', label: t('audit.tab.logs'), icon: <ScrollText size={14} /> },
  ];

  return (
    <PageBase
      title={t('audit.title')}
      subtitle={t('audit.subtitle')}
      icon={<Activity size={22} strokeWidth={1.8} />}
      extra={
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5 rounded-lg border border-black/[0.05] bg-white px-3 py-1.5 shadow-sm">
            <StatusDot variant="green" />
            <code className="text-[length:var(--text-xs)] font-mono text-black/50">{rootPath}</code>
          </div>
        </div>
      }
    >
      <div className="space-y-6">
        <div className="max-w-4xl">
          <SegmentedControl segments={segments} active={activeTab} onChange={setActiveTab} />
        </div>

        <div className="min-h-[500px] animate-in fade-in slide-in-from-bottom-2 duration-500">
          {activeTab === 'overview' && (
            <AuditOverview
              fsTree={fsTree}
              databases={databases}
              dbCount={databases.length}
              totalRows={totalRows}
              logCount={logs.length}
              usersCount={users.length}
              latestLogAt={logs[0]?.created_at ?? null}
            />
          )}

          {activeTab === 'analytics' && (
            <AnalyticsTab analytics={analytics} databases={databases} users={users} />
          )}
          {activeTab === 'users' && <UsersTab users={users} />}
          {activeTab === 'fs' && <FsTab fsTree={fsTree} configPath={configPath} />}
          {activeTab === 'db' && <DbTab databases={databases} />}
          {activeTab === 'logs' && <LogsTab logs={logs} appLogs={appLogs} />}
        </div>
      </div>
    </PageBase>
  );
}
