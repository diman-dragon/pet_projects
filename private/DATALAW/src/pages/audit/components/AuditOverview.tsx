// src/pages/audit/components/AuditOverview.tsx

import { Activity, Clock, Database, FolderTree, HardDrive, Server, Users } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { DbInfo } from '../types';

interface AuditOverviewProps {
  fsTree: any;
  databases: DbInfo[];
  dbCount: number;
  totalRows: number;
  logCount: number;
  usersCount: number;
  latestLogAt?: string | null;
}

export function AuditOverview({
  fsTree,
  databases,
  dbCount,
  totalRows,
  logCount,
  usersCount,
  latestLogAt,
}: AuditOverviewProps) {
  const { t } = useI18n();

  const countFiles = (node: any): number => {
    if (!node) return 0;
    if (node.isFile) return 1;
    return node.children?.reduce((acc: number, child: any) => acc + countFiles(child), 0) || 0;
  };

  const countFolders = (node: any): number => {
    if (!node) return 0;
    if (!node.isFile) {
      return (
        1 + (node.children?.reduce((acc: number, child: any) => acc + countFolders(child), 0) || 0)
      );
    }
    return 0;
  };

  const fileCount = countFiles(fsTree);
  const folderCount = countFolders(fsTree);

  const stats = [
    {
      icon: <FolderTree size={18} />,
      label: t('audit.overview.fs'),
      value: fsTree
        ? t('audit.overview.fsSummary', { folders: folderCount, files: fileCount })
        : t('audit.overview.unavailable'),
    },
    { icon: <Database size={18} />, label: t('audit.overview.db'), value: String(dbCount) },
    {
      icon: <Server size={18} />,
      label: t('audit.overview.rows'),
      value: totalRows.toLocaleString('ru-RU'),
    },
    { icon: <Activity size={18} />, label: t('audit.overview.logs'), value: String(logCount) },
    { icon: <Users size={18} />, label: t('audit.overview.staff'), value: String(usersCount) },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {stats.map((stat, idx) => (
          <div
            key={idx}
            className="group rounded-xl border border-black/[0.06] bg-white p-5 shadow-sm transition-all duration-200 hover:shadow-md"
          >
            <div className="flex size-10 items-center justify-center rounded-lg bg-black/[0.03] text-black/40">
              {stat.icon}
            </div>
            <div className="mt-4">
              <p className="text-[length:var(--text-xs)] font-bold uppercase tracking-wider text-black/35">
                {stat.label}
              </p>
              <p className="mt-1 text-[length:var(--text-md)] font-bold text-black/80">
                {stat.value}
              </p>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-blue-100/50 bg-gradient-to-br from-blue-50/30 to-white p-5">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex size-8 items-center justify-center rounded-lg bg-blue-100">
              <HardDrive size={14} className="text-blue-600" />
            </div>
            <span className="text-[length:var(--text-xs)] font-bold uppercase tracking-wider text-blue-600/60">
              {t('audit.overview.kernel')}
            </span>
          </div>
          <p className="text-[length:var(--text-sm)] leading-relaxed text-black/60">
            {t('audit.overview.kernelText', {
              databases: databases.map((db) => db.key).join(', '),
              tables: databases.reduce((acc, db) => acc + db.tables.length, 0),
            })}
          </p>
        </div>

        <div className="rounded-xl border border-emerald-100/50 bg-gradient-to-br from-emerald-50/30 to-white p-5">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex size-8 items-center justify-center rounded-lg bg-emerald-100">
              <Clock size={14} className="text-emerald-600" />
            </div>
            <span className="text-[length:var(--text-xs)] font-bold uppercase tracking-wider text-emerald-600/60">
              {t('audit.overview.activity')}
            </span>
          </div>
          <p className="text-[length:var(--text-sm)] leading-relaxed text-black/60">
            {t('audit.overview.lastLog')}{' '}
            {latestLogAt ? (
              <span className="font-mono text-black/80">
                {new Date(latestLogAt).toLocaleString('ru-RU')}
              </span>
            ) : (
              t('audit.overview.noLogs')
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
