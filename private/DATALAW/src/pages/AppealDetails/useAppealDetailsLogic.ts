import { useState, useEffect, useCallback, useRef } from 'react';
import { open as dialogOpen } from '@tauri-apps/plugin-dialog';
import { listen } from '@tauri-apps/api/event';

import { businessApi, type AppealWorkbenchPayload } from '@/services/backend/businessApi';

import { ShellService } from '../../services/shell/ShellService';
import type { Appeal, AppealDocument } from '../../types/entities';
import type { AppealSummary, DeadlineEvent, ScannedFile } from './types';
import { createLogger } from '../../services/logger/Logger';

const log = createLogger('AppealDetailsLogic');

interface ChainContext {
  lineId: number;
  caseId: number;
  appealTitle: string;
  lineTitle: string;
  caseTitle: string;
}

export interface UploadState {
  active: boolean;
  stage: 'idle' | 'hashing' | 'ai' | 'saving' | 'chain' | 'done' | 'error';
  fileName: string;
  aiToken: string;
  progress?: string;
  error?: string;
}

const IDLE_UPLOAD: UploadState = {
  active: false,
  stage: 'idle',
  fileName: '',
  aiToken: '',
};

function mapWorkbench(payload: AppealWorkbenchPayload) {
  const files: ScannedFile[] = payload.files.map((file) => ({
    ...file,
    docType: file.doc_type,
    dateStamp: file.date_stamp,
    parsedDate: file.parsed_date ? new Date(file.parsed_date) : null,
    isStandard: file.is_standard,
    description: file.description,
    version: file.version,
    status: file.status,
  }));
  const deadlines: DeadlineEvent[] = payload.deadlines.map((event) => ({
    ...event,
    date: new Date(event.date),
    labelKey: event.label_key,
    isPast: event.is_past,
    isToday: event.is_today,
    isUrgent: event.is_urgent,
    filePath: event.file_path ?? undefined,
    fileName: event.file_name ?? undefined,
  }));
  const docMap = new Map<string, AppealDocument>();
  payload.details.documents.forEach((doc) => {
    docMap.set(doc.file_path, doc);
  });
  log.info('Mapped workbench', {
    docsCount: payload.details.documents.length,
    hasGeneratedSummary: payload.details.summary !== null,
  });
  return {
    appeal: payload.details.appeal as Appeal,
    summary: payload.details.summary as AppealSummary | null,
    folderPath: payload.details.folder_path,
    chainCtx: payload.details.chain_context
      ? {
        lineId: payload.details.chain_context.line_id,
        caseId: payload.details.chain_context.case_id,
        appealTitle: payload.details.appeal.title,
        lineTitle: payload.details.chain_context.line_title,
        caseTitle: payload.details.chain_context.case_title,
      }
      : null,
    files,
    deadlines,
    pendingRetro: payload.pending_retro,
    docMap,
  };
}

export function useAppealDetailsLogic(appealId: number | null) {
  const [appeal, setAppeal] = useState<Appeal | null>(null);
  const [summary, setSummary] = useState<AppealSummary | null>(null);
  const [folderPath, setFolderPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [errorDetail, setErrorDetail] = useState<string | null>(null);
  const [uploadState, setUploadState] = useState<UploadState>(IDLE_UPLOAD);
  const [pendingRetro, setPendingRetro] = useState(false);
  const [chainCtx, setChainCtx] = useState<ChainContext | null>(null);
  const [chainSummaryStatus, setChainSummaryStatus] = useState<'idle' | 'processing' | 'completed' | 'error'>('idle');
  const [docMap, setDocMap] = useState<Map<string, AppealDocument>>(new Map());
  const [files, setFiles] = useState<ScannedFile[]>([]);
  const [deadlines, setDeadlines] = useState<DeadlineEvent[]>([]);
  const [lastScanned, setLastScanned] = useState<Date | null>(null);
  const [scanLoading, setScanLoading] = useState(false);

  // Ref to track pending reset timers so they can be cancelled on unmount.
  const resetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cancel any pending reset timer when the component unmounts.
  useEffect(() => {
    return () => {
      if (resetTimerRef.current !== null) {
        clearTimeout(resetTimerRef.current);
      }
    };
  }, []);

  const applyWorkbench = useCallback((payload: AppealWorkbenchPayload) => {
    const mapped = mapWorkbench(payload);
    setAppeal(mapped.appeal);
    setSummary(mapped.summary);
    setFolderPath(mapped.folderPath);
    setChainCtx(mapped.chainCtx);
    setFiles(mapped.files);
    setDeadlines(mapped.deadlines);
    setPendingRetro(mapped.pendingRetro);
    setDocMap(mapped.docMap);
    setLastScanned(new Date(payload.last_scanned_at));
  }, []);

  const loadData = useCallback(async () => {
    if (!appealId) return;
    setLoading(true);
    setError(null);
    try {
      const payload = await businessApi.getAppealWorkbench(appealId);
      applyWorkbench(payload);
    } catch (err) {
      const typed = err as { key?: string; message?: string };
      log.error('loadData error', typed);
      setError(typed.key || typed.message || 'errors.common.unknown');
      setErrorDetail(typed.message ?? null);
    } finally {
      setLoading(false);
    }
  }, [appealId, applyWorkbench]);

  const rescan = useCallback(async () => {
    if (!appealId) return;
    log.info('Starting rescan for appeal', { appealId });
    setScanLoading(true);
    try {
      const payload = await businessApi.getAppealWorkbench(appealId);
      log.info('Workbench payload received', { documentsCount: payload.details.documents.length });
      applyWorkbench(payload);
    } finally {
      setScanLoading(false);
    }
  }, [appealId, applyWorkbench]);

  const handleOpenFile = useCallback(async (path: string) => {
    try {
      await ShellService.open(path);
    } catch (e) {
      log.error('Shell open failed', e);
    }
  }, []);

  const handleOpenFolder = useCallback(async () => {
    if (!folderPath) return;
    try {
      await ShellService.open(folderPath);
    } catch (e) {
      log.error('Folder open failed', e);
    }
  }, [folderPath]);

  const handleRenameToStandard = useCallback(
    async (file: ScannedFile, newStem: string) => {
      try {
        await businessApi.renameAppealFile(file.path, newStem);
        await rescan();
        return true;
      } catch (e) {
        log.error('Rename failed', e);
        return false;
      }
    },
    [rescan],
  );

  const handleUpload = useCallback(async () => {
    if (!appealId) return;
    const selected = await dialogOpen({
      multiple: false,
      filters: [
        {
          name: 'documents',
          extensions: ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'txt', 'jpg', 'jpeg', 'png', 'webp'],
        },
      ],
    });
    if (!selected || typeof selected !== 'string') return;
    const originalName = selected.split('/').pop()?.split('\\').pop() ?? 'document';
    setUploadState({ active: true, stage: 'hashing', fileName: originalName, aiToken: '' });
    try {
      const payload = await businessApi.uploadAppealDocument(appealId, selected);
      setUploadState((current) => ({ ...current, stage: 'done' }));
      applyWorkbench(payload);
      if (resetTimerRef.current !== null) clearTimeout(resetTimerRef.current);
      resetTimerRef.current = setTimeout(() => setUploadState(IDLE_UPLOAD), 1500);
    } catch (e) {
      const typed = e as { key?: string; message?: string };
      setUploadState((current) => ({
        ...current,
        stage: 'error',
        error: typed.key || typed.message || 'documentUpload.error',
      }));
      if (resetTimerRef.current !== null) clearTimeout(resetTimerRef.current);
      resetTimerRef.current = setTimeout(() => setUploadState(IDLE_UPLOAD), 2500);
    }
  }, [appealId, applyWorkbench]);

  const handleRetroactive = useCallback(async () => {
    if (!appealId) return;
    setUploadState({
      active: true,
      stage: 'ai',
      fileName: 'documentUpload.retroactive',
      aiToken: '',
    });
    try {
      const payload = await businessApi.retroProcessAppealDocuments(appealId);
      setUploadState((current) => ({ ...current, stage: 'done' }));
      applyWorkbench(payload);
      if (resetTimerRef.current !== null) clearTimeout(resetTimerRef.current);
      resetTimerRef.current = setTimeout(() => setUploadState(IDLE_UPLOAD), 1500);
    } catch (e) {
      const typed = e as { key?: string; message?: string };
      setUploadState((current) => ({
        ...current,
        stage: 'error',
        error: typed.key || typed.message || 'documentUpload.error',
      }));
      if (resetTimerRef.current !== null) clearTimeout(resetTimerRef.current);
      resetTimerRef.current = setTimeout(() => setUploadState(IDLE_UPLOAD), 2500);
    }
  }, [appealId, applyWorkbench]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    let canceled = false;
    let unlistenAnalysis: (() => void) | undefined;
    let unlistenChain: (() => void) | undefined;

    listen<{ appeal_id: number }>('analysis-completed', (event) => {
      log.info('analysis-completed event received', { appealId: event.payload.appeal_id, currentAppealId: appealId });
      if (event.payload.appeal_id === appealId) {
        log.info('Triggering rescan for appeal', { appealId });
        setChainSummaryStatus('processing');
        rescan();
      }
    }).then((fn) => {
      if (canceled) {
        fn();
      } else {
        unlistenAnalysis = fn;
      }
    });

    listen<{ appeal_id: number }>('chain-summary-completed', (event) => {
      log.info('chain-summary-completed event received', { appealId: event.payload.appeal_id, currentAppealId: appealId });
      if (event.payload.appeal_id === appealId) {
        log.info('Chain summary completed, triggering rescan', { appealId });
        setChainSummaryStatus('completed');
        rescan();
      }
    }).then((fn) => {
      if (canceled) {
        fn();
      } else {
        unlistenChain = fn;
      }
    });

    return () => {
      canceled = true;
      unlistenAnalysis?.();
      unlistenChain?.();
    };
  }, [appealId, rescan]);

  return {
    appeal,
    summary,
    folderPath,
    loading,
    error,
    errorDetail,
    folderScan: {
      files,
      loading: scanLoading,
      error: null,
      lastScanned,
      rescan,
    },
    deadlines,
    uploadState,
    pendingRetro,
    docMap,
    chainCtx,
    chainSummaryStatus,
    loadData,
    handleOpenFile,
    handleOpenFolder,
    handleRenameToStandard,
    handleUpload,
    handleRetroactive,
  };
}
