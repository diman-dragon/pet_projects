// src/pages/audit/types.ts

import type { AuditEntryRecord } from '../../services/audit/AuditService';
import type { UserRecord as AuthUserRecord } from '../../services/auth/UserService';
import type { TableStructure } from '../../services/db/InspectorService';
import type { FileNode } from '../../services/fs/ScannerService';
import type { AppLogRecord } from '../../services/logger/LogHistoryService';

export interface TableExtended extends TableStructure {
  rowCount: number;
}

export interface DbInfo {
  key: string;
  tables: TableExtended[];
}

export type UserRecord = AuthUserRecord;

export interface MonthStat {
  month: string;
  count: number;
}

export interface CaseTypeStat {
  case_type: string | null;
  count: number;
}

export interface AuditAnalytics {
  clientsByMonth: MonthStat[];
  casesByType: CaseTypeStat[];
}

export interface AuditData {
  fsTree: FileNode | null;
  databases: DbInfo[];
  logs: AuditEntryRecord[];
  appLogs: AppLogRecord[];
  users: UserRecord[];
  analytics: AuditAnalytics;
  configPath: string;
  loading: boolean;
}
