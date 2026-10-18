// src/pages/audit/useAuditLogic.ts

import { useEffect, useState } from 'react';
import { appConfigDir } from '@tauri-apps/api/path';
import { AuditService } from '../../services/audit/AuditService';
import { UserService } from '../../services/auth/UserService';
import { ConnectionManager } from '../../services/db/ConnectionManager';
import { InspectorService } from '../../services/db/InspectorService';
import { ScannerService } from '../../services/fs/ScannerService';
import { LogHistoryService } from '../../services/logger/LogHistoryService';
import type { AuditData, CaseTypeStat, MonthStat } from './types';

const CLIENTS_DB = 'clients_main';

async function loadUsers() {
  const profiles = await UserService.listProfiles();
  const records = await Promise.all(profiles.map((item) => UserService.getById(item.id)));
  return records.filter((record): record is NonNullable<typeof record> => record !== null);
}

async function loadClientsByMonth(): Promise<MonthStat[]> {
  if (!ConnectionManager.exists(CLIENTS_DB)) return [];

  const db = ConnectionManager.get(CLIENTS_DB);
  return db.select<MonthStat[]>(
    `SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as count
     FROM clients
     WHERE created_at IS NOT NULL
     GROUP BY strftime('%Y-%m', created_at)
     ORDER BY month ASC`,
  );
}

async function loadCasesByType(): Promise<CaseTypeStat[]> {
  if (!ConnectionManager.exists(CLIENTS_DB)) return [];

  const db = ConnectionManager.get(CLIENTS_DB);
  return db.select<CaseTypeStat[]>(
    `SELECT NULLIF(TRIM(case_type), '') as case_type, COUNT(*) as count
     FROM cases
     GROUP BY NULLIF(TRIM(case_type), '')
     ORDER BY count DESC, case_type ASC`,
  );
}

export function useAuditLogic(rootPath: string) {
  const [data, setData] = useState<AuditData>({
    fsTree: null,
    databases: [],
    logs: [],
    appLogs: [],
    users: [],
    analytics: {
      clientsByMonth: [],
      casesByType: [],
    },
    configPath: '',
    loading: true,
  });

  useEffect(() => {
    async function run() {
      try {
        const [configPath, fsTree, logs, appLogs, users, clientsByMonth, casesByType] = await Promise.all([
          appConfigDir(),
          ScannerService.scan(rootPath),
          AuditService.listAllEntries(),
          LogHistoryService.getAllLogs(),
          loadUsers(),
          loadClientsByMonth(),
          loadCasesByType(),
        ]);

        const keys = ['user_system', 'kb_main', 'clients_main'];
        const databases = await Promise.all(
          keys.map(async (key) => ({
            key,
            tables: await InspectorService.getDbStructureWithCounts(key),
          })),
        );

        setData({
          fsTree,
          databases,
          logs,
          appLogs,
          users,
          analytics: {
            clientsByMonth,
            casesByType,
          },
          configPath,
          loading: false,
        });
      } catch (error) {
        console.error(error);
        setData((previous) => ({ ...previous, loading: false }));
      }
    }

    run();
  }, [rootPath]);

  return data;
}
