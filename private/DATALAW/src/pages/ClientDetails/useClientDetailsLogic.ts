// src/pages/ClientDetails/useClientDetailsLogic.ts
import { useState, useEffect, useCallback, useMemo } from 'react';
import { listen } from '@tauri-apps/api/event';
import { businessApi } from '@/services/backend/businessApi';
import { useOllamaStore } from '@/store/ollamaStore';
import { createLogger } from '@/services/logger/Logger';
import { ID } from '@/types/entities';
import type { Case, ClientFull, CaseStatus } from './types';
import { RiskScannerAgent, type DetectedRisk, type RiskScanResult } from '@/services/ai/agents/RiskScannerAgent';

export type DetailsView = 'cases' | 'info';

const DEFAULT_CASE: Partial<Case> = {
  title: '',
  status: 'ACTIVE' as CaseStatus,
  case_type: 'litigation',
  litigation_subtype: 'civil',
  counterparty: '',
  case_number: '',
  metadata: '',
  summary: '',
};

export function useClientDetailsLogic(clientId: ID) {
  const log = createLogger('useClientDetailsLogic');
  const [client, setClient] = useState<ClientFull | null>(null);
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<DetailsView>('cases');
  const [editingCase, setEditingCase] = useState<Case | null>(null);
  const [isNew, setIsNew] = useState(false);
  const [formData, setFormData] = useState<Partial<Case>>({});
  const [isSaving, setIsSaving] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Case | null>(null);

  // [0.9] Состояние генерации сводки клиента
  const [isSummaryGenerating, setIsSummaryGenerating] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [isRiskScanning, setIsRiskScanning] = useState(false);
  const [riskScanResult, setRiskScanResult] = useState<RiskScanResult | null>(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [sortType, setSortType] = useState('date_desc');

  const numericId = Number(clientId);
  const sidecarStatus = useOllamaStore((s) => s.sidecarStatus);
  const aiAvailable = sidecarStatus.running;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const payload = await businessApi.getClientDetails(numericId);
      const rawClient = payload.client;
      const mappedClient: ClientFull = {
        ...rawClient,
        type: String(rawClient.type) === 'IP' || String(rawClient.type) === 'INDIVIDUAL'
          ? 'INDIVIDUAL'
          : 'LEGAL',
        client_data: rawClient.client_data ?? rawClient.metadata ?? null,
      };
      setClient(mappedClient);
      setCases(payload.cases as Case[]);
    } catch (err) {
      log.error('[ClientDetails] load error:', err);
    } finally {
      setLoading(false);
    }
  }, [numericId]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    let canceled = false;
    let unlisten: (() => void) | undefined;

    listen<{ type: string; id: number; summary: string }>('summary-completed', (event) => {
      if (event.payload.type !== 'client' || event.payload.id !== numericId) return;
      setClient((prev) => (prev ? { ...prev, summary: event.payload.summary } : prev));
      setIsSummaryGenerating(false);
    }).then((fn) => {
      if (canceled) {
        fn();
      } else {
        unlisten = fn;
      }
    });

    return () => {
      canceled = true;
      unlisten?.();
    };
  }, [numericId]);

  // [0.9] Авто-запрос сводки при монтировании если её нет и AI доступен
  useEffect(() => {
    if (!client || !aiAvailable || isSummaryGenerating) return;
    if (client.summary) return; // сводка уже есть

    setIsSummaryGenerating(true);
    setSummaryError(null);

    businessApi
      .generateClientSummary(numericId)
      .then(() => {
        // Перезагружаем данные клиента чтобы подтянуть обновлённую сводку
        return load();
      })
      .catch((err) => {
        log.warn('[ClientDetails] auto-summary failed:', err);
        setSummaryError(err?.message ?? 'Не удалось сгенерировать сводку');
      })
      .finally(() => {
        setIsSummaryGenerating(false);
      });
  }, [client?.id, aiAvailable]); // eslint-disable-line react-hooks/exhaustive-deps

  // [0.9] Ручной запуск генерации сводки
  const generateSummary = useCallback(async () => {
    if (!client || isSummaryGenerating) return;
    setIsSummaryGenerating(true);
    setSummaryError(null);
    try {
      await businessApi.generateClientSummary(numericId);
      await load();
    } catch (err: any) {
      setSummaryError(err?.message ?? 'Не удалось сгенерировать сводку');
    } finally {
      setIsSummaryGenerating(false);
    }
  }, [client, numericId, isSummaryGenerating, load]);

  const criticalRisksCount = useMemo(
    () => riskScanResult?.risks.filter((risk) => risk.severity === 'critical').length ?? 0,
    [riskScanResult],
  );

  const groupedRisks = useMemo(() => {
    const empty = { critical: [] as DetectedRisk[], warnings: [] as DetectedRisk[], ok: [] as string[] };
    if (!riskScanResult) return empty;

    const critical = riskScanResult.risks.filter((risk) => risk.severity === 'critical');
    const warnings = riskScanResult.risks.filter((risk) => risk.severity !== 'critical');
    const ok = riskScanResult.risks.length === 0 ? ['Критических рисков не обнаружено'] : [];
    return { critical, warnings, ok };
  }, [riskScanResult]);

  const scanRisks = useCallback(async () => {
    setIsRiskScanning(true);
    try {
      const result = await RiskScannerAgent.scan({
        cases: cases.map((item) => ({
          id: Number(item.id),
          name: item.title,
          summary: item.summary ?? null,
          status: item.status,
        })),
      });
      setRiskScanResult(result);
    } finally {
      setIsRiskScanning(false);
    }
  }, [cases]);

  const openCreate = () => {
    setIsNew(true);
    setEditingCase(null);
    setFormData({ ...DEFAULT_CASE });
    setIsEditOpen(true);
  };

  const openEdit = (c: Case) => {
    setIsNew(false);
    setEditingCase(c);
    setFormData({ ...c });
    setIsEditOpen(true);
  };

  const closeModal = () => {
    setEditingCase(null);
    setIsNew(false);
    setFormData({});
    setIsEditOpen(false);
  };

  const handleSaveCase = async () => {
    let titleToSave = formData.title?.trim() || '';

    if (!titleToSave && formData.case_type === 'contract' && formData.counterparty) {
      const year = formData.metadata || String(new Date().getFullYear());
      titleToSave = `${year} - ${formData.counterparty}`;
    }

    if (!titleToSave && formData.case_type === 'consulting' && formData.metadata) {
      const [year, month] = formData.metadata.split('|');
      titleToSave = `${String(month).padStart(2, '0')}/${year} - Консультирование`;
    }

    if (!titleToSave || !client) return;

    setIsSaving(true);
    const cleanStatus = String(formData.status || 'ACTIVE')
      .trim()
      .toUpperCase() as CaseStatus;
    try {
      if (isNew) {
        await businessApi.saveCase({
          client_id: numericId,
          client_folder_name: client.folder_name || '',
          title: titleToSave,
          status: cleanStatus,
          case_type: formData.case_type || '',
          litigation_subtype: formData.litigation_subtype || undefined,
          counterparty: formData.counterparty || undefined,
          case_number: formData.case_number || '',
          metadata: formData.metadata || '',
          summary: formData.summary || '',
          contract_date: formData.contract_date || undefined,
          contract_type: formData.contract_type || undefined,
          contract_amount: formData.contract_amount || undefined,
          metadata_description: formData.metadata_description || undefined,
        });
      } else if (editingCase) {
        await businessApi.saveCase({
          id: Number(editingCase.id),
          client_id: numericId,
          client_folder_name: client.folder_name || '',
          title: titleToSave,
          status: cleanStatus,
          case_type: formData.case_type || '',
          litigation_subtype: formData.litigation_subtype || undefined,
          counterparty: formData.counterparty || undefined,
          case_number: formData.case_number || '',
          metadata: formData.metadata || '',
          summary: formData.summary || '',
          contract_date: formData.contract_date || undefined,
          contract_type: formData.contract_type || undefined,
          contract_amount: formData.contract_amount || undefined,
          metadata_description: formData.metadata_description || undefined,
        });
      }
      await load();
      closeModal();
    } catch (err) {
      log.error('[ClientDetails] save error:', err);
    } finally {
      setIsSaving(false);
    }
  };

  const confirmDelete = (c: Case) => setDeleteTarget(c);

  const handleDeleteCase = async () => {
    if (!deleteTarget) return;
    try {
      await businessApi.deleteCase(Number(deleteTarget.id));
      await load();
    } catch (err) {
      log.error('[ClientDetails] delete error:', err);
    } finally {
      setDeleteTarget(null);
    }
  };

  const filteredAndSortedCases = useMemo(() => {
    let result = [...cases];

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      result = result.filter(
        (c) =>
          c.title.toLowerCase().includes(q) ||
          c.case_number?.toLowerCase().includes(q) ||
          c.summary?.toLowerCase().includes(q),
      );
    }

    if (filterType !== 'all') {
      if (filterType.startsWith('litigation_')) {
        const subtype = filterType.split('_')[1];
        result = result.filter((c) => c.case_type === 'litigation' && c.litigation_subtype === subtype);
      } else {
        result = result.filter((c) => c.case_type === filterType);
      }
    }

    if (filterStatus !== 'all') {
      result = result.filter((c) => c.status === filterStatus);
    }

    result.sort((a, b) => {
      switch (sortType) {
        case 'date_asc':
          return (a.created_at || '').localeCompare(b.created_at || '');
        case 'date_desc':
          return (b.created_at || '').localeCompare(a.created_at || '');
        case 'name_asc':
          return a.title.localeCompare(b.title);
        case 'name_desc':
          return b.title.localeCompare(a.title);
        default:
          return 0;
      }
    });

    return result;
  }, [cases, searchQuery, filterType, filterStatus, sortType]);

  return {
    client,
    cases: filteredAndSortedCases,
    loading,
    view,
    setView,
    editingCase,
    isNew,
    isEditOpen,
    setIsEditOpen,
    formData,
    setFormData,
    isSaving,
    deleteTarget,
    setDeleteTarget,
    openCreate,
    openEdit,
    closeModal,
    handleSaveCase,
    confirmDelete,
    handleDeleteCase,
    searchQuery,
    setSearchQuery,
    filterType,
    setFilterType,
    filterStatus,
    setFilterStatus,
    sortType,
    setSortType,
    // [0.9] сводка
    isSummaryGenerating,
    summaryError,
    generateSummary,
    isRiskScanning,
    riskScanResult,
    groupedRisks,
    criticalRisksCount,
    scanRisks,
  };
}
