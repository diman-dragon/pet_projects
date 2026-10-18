import type {
  AppealSummaryPayload,
  DeadlineEventPayload,
  ScannedFilePayload,
} from '@/services/backend/businessApi';

export type AppealView = 'documents' | 'timeline' | 'summary';

export type AppealSummary = AppealSummaryPayload;

export interface ScannedFile extends Omit<
  ScannedFilePayload,
  'parsed_date' | 'date_stamp' | 'doc_type' | 'is_standard'
> {
  docType: string | null;
  dateStamp: string | null;
  parsedDate: Date | null;
  isStandard: boolean;
}

export interface DeadlineEvent extends Omit<
  DeadlineEventPayload,
  'date' | 'label_key' | 'is_past' | 'is_today' | 'is_urgent' | 'file_path' | 'file_name'
> {
  date: Date;
  labelKey: string;
  isPast: boolean;
  isToday: boolean;
  isUrgent: boolean;
  filePath?: string;
  fileName?: string;
}
