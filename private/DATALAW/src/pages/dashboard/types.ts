// src/pages/dashboard/types.ts

export type CalendarValue = Date | null | [Date | null, Date | null];

export type DeadlineType = 'court' | 'contract' | 'tax';

export interface DeadlineItem {
  id: number;
  title: string;
  date: string;
  type: DeadlineType;
}

export interface LogItem {
  id: number;
  time: string;
  user: string;
  msg: string;
}

export interface StatData {
  docsCount: number;
  clientsCount: number;
  winRate: number;
  pnpProgress: number;
  clientDist: { name: string; value: number; color: string }[];
  casesFlow: { month: string; corporate: number; private: number }[];
  deadlines: DeadlineItem[];
  recentLogs: LogItem[];
}
