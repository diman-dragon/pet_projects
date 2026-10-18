import { useCallback, useEffect, useState } from 'react';

import { businessApi, type AppErrorLike } from '@/services/backend/businessApi';
import type { Appeal, Line, Case } from '@/types/entities';

export type DetailsView = 'appeals' | 'info';

const DEFAULT_APPEAL: Partial<Appeal> = {
  title: '',
  status: 'DRAFT',
  agency_code: '',
  incoming_number: '',
  outgoing_number: '',
  appeal_date: '',
};

type LineWithPath = Omit<Line, 'description'> & {
  description: string | null;
  case_folder: string;
  client_folder: string;
};

export function useLineDetailsLogic(lineId: number | null) {
  const [line, setLine] = useState<LineWithPath | null>(null);
  const [case_, setCase] = useState<Case | null>(null);
  const [appeals, setAppeals] = useState<Appeal[]>([]);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<DetailsView>('appeals');
  const [editingAppeal, setEditingAppeal] = useState<Appeal | null>(null);
  const [isNew, setIsNew] = useState(false);
  const [formData, setFormData] = useState<Partial<Appeal>>({ ...DEFAULT_APPEAL });
  const [isSaving, setIsSaving] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Appeal | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!lineId) return;
    setLoading(true);
    setError(null);

    try {
      const payload = await businessApi.getLineDetails(lineId);
      setLine({
        ...payload.line,
        description: payload.line.description ?? null,
      });
      setAppeals(payload.appeals);

      // Fetch case details to know the scenario
      if (payload.line.case_id) {
        const casePayload = await businessApi.getCaseDetails(payload.line.case_id);
        setCase(casePayload.case_item);
      }
    } catch (err) {
      console.error('[LineDetails] load error:', err);
      setError((err as AppErrorLike).key || 'line.error.load');
    } finally {
      setLoading(false);
    }
  }, [lineId]);

  useEffect(() => {
    if (lineId) load();
  }, [load, lineId]);

  const openCreate = () => {
    setIsNew(true);
    setFormData({ ...DEFAULT_APPEAL });
    setEditingAppeal(null);
    setError(null);
  };

  const openEdit = (appeal: Appeal) => {
    setIsNew(false);
    setEditingAppeal(appeal);
    setFormData({ ...appeal });
    setError(null);
  };

  const closeModal = () => {
    setEditingAppeal(null);
    setIsNew(false);
    setFormData({ ...DEFAULT_APPEAL });
    setError(null);
  };

  const handleSaveAppeal = async () => {
    if (!formData.title?.trim()) {
      setError('line.error.titleRequired');
      return;
    }
    if (!line) {
      setError('line.error.lineNotLoaded');
      return;
    }

    setIsSaving(true);
    setError(null);

    try {
      const payload = {
        id: editingAppeal ? Number(editingAppeal.id) : undefined,
        line_id: lineId!,
        client_folder_name: line.client_folder,
        case_folder_name: line.case_folder,
        line_folder_name: line.folder_name,
        case_type: case_?.case_type,
        title: formData.title,
        status: formData.status ?? 'DRAFT',
        agency_code: formData.agency_code ?? undefined,
        appeal_date: formData.appeal_date ?? undefined,
        incoming_number: formData.incoming_number ?? undefined,
        outgoing_number: formData.outgoing_number ?? undefined,
      };

      await businessApi.saveAppeal(payload);
      await load();
      closeModal();
    } catch (err) {
      console.error('[LineDetails] save error:', err);
      setError((err as AppErrorLike).key || 'line.error.save');
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteAppeal = async () => {
    if (!deleteTarget?.id) return;

    try {
      await businessApi.deleteAppeal(Number(deleteTarget.id));
      await load();
    } catch (err) {
      console.error('[LineDetails] delete error:', err);
      setError('line.error.delete');
    } finally {
      setDeleteTarget(null);
    }
  };

  const handleUpdateLine = async (updates: Partial<Line>) => {
    if (!line || !case_) return;
    setIsSaving(true);
    try {
      await businessApi.saveLine({
        id: line.id,
        case_id: line.case_id,
        case_folder_name: line.case_folder,
        client_folder_name: line.client_folder,
        case_type: case_.case_type,
        title: updates.title || line.title,
        status: updates.status || line.status || 'ACTIVE',
        description: updates.description || line.description || '',
        consultation_text: updates.consultation_text !== undefined ? updates.consultation_text : line.consultation_text,
        consultation_topic: updates.consultation_topic !== undefined ? updates.consultation_topic : line.consultation_topic,
        contract_analysis: updates.contract_analysis !== undefined ? updates.contract_analysis : line.contract_analysis,
        contract_risks: updates.contract_risks !== undefined ? updates.contract_risks : line.contract_risks,
      } as any);
      await load();
    } catch (err) {
      console.error('[LineDetails] update line error:', err);
      setError('line.error.save');
    } finally {
      setIsSaving(false);
    }
  };

  return {
    line,
    case_,
    appeals,
    loading,
    view,
    setView,
    editingAppeal,
    isNew,
    formData,
    setFormData,
    isSaving,
    deleteTarget,
    setDeleteTarget,
    error,
    openCreate,
    openEdit,
    closeModal,
    handleSaveAppeal,
    handleDeleteAppeal,
    handleUpdateLine,
  };
}
