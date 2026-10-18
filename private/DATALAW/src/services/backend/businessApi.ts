import { invoke } from '@tauri-apps/api/core';

import type { Appeal, AppealDocument, Case, Client, Line } from '@/types/entities';
import type { StatData } from '@/pages/dashboard/types';
import { useOllamaStore } from '@/store/ollamaStore';
import { getContextParams } from '@/services/ai/ModelContextProfile';

export interface AppErrorLike {
  key?: string;
  message?: string;
}

export type TrashEntityType = 'client' | 'case' | 'line' | 'appeal';

export interface LineDetails extends Line {
  case_folder: string;
  client_folder: string;
}

export interface ClientDetailsPayload {
  client: Client;
  cases: Case[];
}

export interface ClientDossierProvider {
  id: string;
  name: string;
  description: string;
  status: string;
  enabled: boolean;
}

export interface ClientDossierSnapshot {
  id: number;
  client_id: number;
  tax_id: string;
  status: string;
  providers_json: string;
  summary_text?: string | null;
  folder_path: string;
  manifest_path: string;
  dossier_json?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface ClientDossierOverviewPayload {
  providers: ClientDossierProvider[];
  snapshots: ClientDossierSnapshot[];
}

export interface RefreshClientDossierPayload {
  snapshot: ClientDossierSnapshot;
  providers: ClientDossierProvider[];
}

export interface CaseDetailsPayload {
  case_item: Case;
  lines: Line[];
}

export interface LineDetailsPayload {
  line: LineDetails;
  appeals: Appeal[];
}

export interface AppealSummaryPayload {
  id: number;
  appeal_id: number;
  text: string;
  key_points?: string[];
  generated_by?: string;
  updated_at?: string;
}

export interface AppealChainContextPayload {
  line_id: number;
  case_id: number;
  line_title: string;
  case_title: string;
}

export interface AppealDetailsPayload {
  appeal: Appeal;
  summary: AppealSummaryPayload | null;
  documents: AppealDocument[];
  folder_path: string | null;
  chain_context: AppealChainContextPayload | null;
}

export interface ScannedFilePayload {
  name: string;
  path: string;
  size: number;
  ext: string;
  doc_type: string | null;
  date_stamp: string | null;
  parsed_date: string | null;
  is_standard: boolean;
  description: string | null;
  version: string | null;
  status: string | null;
}

export interface DeadlineEventPayload {
  id: string;
  date: string;
  label_key: string;
  kind: string;
  is_past: boolean;
  is_today: boolean;
  is_urgent: boolean;
  file_path?: string | null;
  file_name?: string | null;
}

export interface AppealWorkbenchPayload {
  details: AppealDetailsPayload;
  files: ScannedFilePayload[];
  deadlines: DeadlineEventPayload[];
  pending_retro: boolean;
  last_scanned_at: string;
}

export interface GenerateSummaryPayload {
  summary: string;
}

export interface SaveClientPayload {
  id?: number;
  full_name: string;
  client_type: string;
  status: string;
  tax_id?: string;
  email?: string;
  metadata?: string;
}

export interface SaveCasePayload {
  id?: number;
  client_id: number;
  client_folder_name: string;
  title: string;
  status?: string;
  case_type?: string;
  litigation_subtype?: string;
  counterparty?: string;
  case_number?: string;
  metadata?: string;
  summary?: string;
  contract_date?: string;
  contract_type?: string;
  contract_amount?: string;
  metadata_description?: string;
}

export interface SaveLinePayload {
  id?: number;
  case_id: number;
  case_folder_name: string;
  client_folder_name: string;
  case_type?: string | null;
  title: string;
  status?: string;
  description?: string;
  consultation_text?: string;
  consultation_topic?: string;
  contract_analysis?: string;
  contract_risks?: string;
}

export interface SaveAppealPayload {
  id?: number;
  line_id: number;
  client_folder_name: string;
  case_folder_name: string;
  line_folder_name: string;
  case_type?: string | null;
  title: string;
  status?: string;
  agency_code?: string;
  appeal_date?: string;
  incoming_number?: string;
  outgoing_number?: string;
  recipient?: string;
}

export interface HierarchyAppealNode {
  id: number;
  title: string;
  folder_name: string;
}

export interface HierarchyLineNode {
  id: number;
  title: string;
  folder_name: string;
  appeals: HierarchyAppealNode[];
}

export interface HierarchyCaseNode {
  id: number;
  title: string;
  folder_name: string;
  status: string;
  case_type?: string | null;
  counterparty?: string | null;
  litigation_subtype?: string | null;
  lines: HierarchyLineNode[];
}

export interface HierarchyClientNode {
  id: number;
  full_name: string;
  folder_name: string;
  cases: HierarchyCaseNode[];
}

export interface TrashItem {
  entity_type: TrashEntityType;
  entity_id: number;
  entity_label: string;
  deleted_at: string;
  folder_name?: string | null;
}

export interface AgencyDictionaryEntry {
  name: string;
  code: string;
  source: string;
}

export interface AgencyCodeSuggestion {
  name: string;
  code: string;
}

export interface SaveAgencyDictionaryPayload {
  name: string;
  code: string;
}

interface SummaryGenerationOptions {
  modelName?: string;
  numCtx?: number;
}

/**
 * [9.1, 9.2] Получить параметры для генерации сводки
 * Использует выбранную модель и профиль машины для num_ctx
 */
function getSummaryGenerationOptions(): SummaryGenerationOptions {
  const { selectedModel } = useOllamaStore.getState();

  // [9.2] Получаем num_ctx из профиля машины для задачи summarize
  // getContextParams берет profile из ollamaStore автоматически
  const contextParams = getContextParams('summarize');

  return {
    modelName: selectedModel ?? undefined,
    numCtx: contextParams.num_ctx,
  };
}

function normalizeError(error: unknown): AppErrorLike {
  if (typeof error === 'object' && error !== null) {
    const typed = error as AppErrorLike;
    return {
      key: typed.key,
      message: typed.message || JSON.stringify(error),
    };
  }
  return { message: String(error) };
}

async function invokeCommand<T>(command: string, payload?: Record<string, unknown>): Promise<T> {
  try {
    return await invoke<T>(command, payload);
  } catch (error) {
    throw normalizeError(error);
  }
}

// [9.5] Семафор параллельности суммаризации
// minimal → 1 поток, остальные → 2 потока одновременно
class SummarySemaphore {
  private running = 0;
  private queue: Array<() => void> = [];

  private get limit(): number {
    const profile = useOllamaStore.getState().machineProfile;
    return profile === 'minimal' ? 1 : 2;
  }

  async acquire(): Promise<void> {
    if (this.running < this.limit) {
      this.running++;
      return;
    }
    return new Promise((resolve) => {
      this.queue.push(() => {
        this.running++;
        resolve();
      });
    });
  }

  release(): void {
    this.running--;
    const next = this.queue.shift();
    if (next) next();
  }

  async run<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

const summarySemaphore = new SummarySemaphore();

export const businessApi = {
  listClients() {
    return invokeCommand<Client[]>('list_clients');
  },
  saveClient(payload: SaveClientPayload) {
    return invokeCommand<void>('save_client', { payload });
  },
  deleteClient(clientId: number) {
    return invokeCommand<void>('delete_client', { clientId });
  },
  getClientDetails(clientId: number) {
    return invokeCommand<ClientDetailsPayload>('get_client_details', { clientId });
  },
  getClientDossier(clientId: number) {
    return invokeCommand<ClientDossierOverviewPayload>('get_client_dossier', { clientId });
  },
  refreshClientDossier(clientId: number, taxId?: string | null) {
    return invokeCommand<RefreshClientDossierPayload>('refresh_client_dossier', {
      request: { clientId, taxId },
    });
  },
  getHierarchyTree() {
    return invokeCommand<HierarchyClientNode[]>('get_hierarchy_tree');
  },
  saveCase(payload: SaveCasePayload) {
    return invokeCommand<void>('save_case', { payload });
  },
  deleteCase(caseId: number) {
    return invokeCommand<void>('delete_case', { caseId });
  },
  getCaseDetails(caseId: number) {
    return invokeCommand<CaseDetailsPayload>('get_case_details', { caseId });
  },
  saveLine(payload: SaveLinePayload) {
    return invokeCommand<void>('save_line', { payload });
  },
  deleteLine(lineId: number) {
    return invokeCommand<void>('delete_line', { lineId });
  },
  getLineDetails(lineId: number) {
    return invokeCommand<LineDetailsPayload>('get_line_details', { lineId });
  },
  saveAppeal(payload: SaveAppealPayload) {
    return invokeCommand<number>('save_appeal', { payload });
  },
  listAgencyDictionary() {
    return invokeCommand<AgencyDictionaryEntry[]>('list_agency_dictionary');
  },
  suggestAgencyCode(name: string) {
    return invokeCommand<AgencyCodeSuggestion>('suggest_agency_code', { name });
  },
  saveAgencyDictionaryEntry(payload: SaveAgencyDictionaryPayload) {
    return invokeCommand<AgencyDictionaryEntry>('save_agency_dictionary_entry', { payload });
  },
  uploadDocument(payload: {
    file_path: string;
    file_name: string;
    client_folder_name: string;
    case_folder_name: string;
    line_folder_name: string;
    appeal_folder_name: string;
    case_type?: string;
  }) {
    return invokeCommand<string>('upload_document', { payload });
  },
  deleteAppeal(appealId: number) {
    return invokeCommand<void>('delete_appeal', { appealId });
  },
  getAppealDetails(appealId: number) {
    return invokeCommand<AppealDetailsPayload>('get_appeal_details', { appealId });
  },
  getAppealWorkbench(appealId: number) {
    return invokeCommand<AppealWorkbenchPayload>('get_appeal_workbench', { appealId });
  },
  renameAppealFile(filePath: string, newStem: string) {
    return invokeCommand<void>('rename_appeal_file', { filePath, newStem });
  },
  uploadAppealDocument(appealId: number, sourcePath: string) {
    return invokeCommand<AppealWorkbenchPayload>('upload_appeal_document', {
      appealId,
      sourcePath,
    });
  },
  retroProcessAppealDocuments(appealId: number) {
    return invokeCommand<AppealWorkbenchPayload>('retro_process_appeal_documents', { appealId });
  },
  reanalyzeDocument(appealId: number, filePath: string) {
    return invokeCommand<AppealWorkbenchPayload>('reanalyze_document', { appealId, filePath });
  },

  // [9.5] Все generate_*_summary оборачиваем в семафор
  generateClientSummary(clientId: number) {
    return summarySemaphore.run(() =>
      invokeCommand<void>('generate_client_summary', {
        clientId,
        options: getSummaryGenerationOptions(),
      }),
    );
  },
  generateCaseSummary(caseId: number) {
    return summarySemaphore.run(() =>
      invokeCommand<void>('generate_case_summary', {
        caseId,
        options: getSummaryGenerationOptions(),
      }),
    );
  },
  generateLineSummary(lineId: number) {
    return summarySemaphore.run(() =>
      invokeCommand<void>('generate_line_summary', {
        lineId,
        options: getSummaryGenerationOptions(),
      }),
    );
  },

  getDashboardStats() {
    return invokeCommand<StatData>('get_dashboard_stats');
  },
  listTrashItems() {
    return invokeCommand<TrashItem[]>('list_trash_items');
  },
  restoreTrashItem(entityType: TrashEntityType, entityId: number) {
    return invokeCommand<void>('restore_trash_item', { entityType, entityId });
  },
  purgeTrashItem(entityType: TrashEntityType, entityId: number) {
    return invokeCommand<void>('purge_trash_item', { entityType, entityId });
  },
};
