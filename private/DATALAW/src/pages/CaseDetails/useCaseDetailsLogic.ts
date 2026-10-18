// src/pages/CaseDetails/useCaseDetailsLogic.ts

import { useState, useEffect, useCallback, useRef } from 'react';
import { businessApi, type AppErrorLike } from '@/services/backend/businessApi';
import { useI18n } from '../../i18n/AppLanguageProvider';
import type { Case, Line } from './types';
import { DocumentClassifierAgent } from '@/services/ai/agents/DocumentClassifierAgent';

export function useCaseDetailsLogic(caseId: number) {
  const { t } = useI18n();

  // ── Основные данные (согласно стандарту useRegistryLogic) ──
  const [case_, setCase] = useState<Case | null>(null);
  const [items, setItems] = useState<Line[]>([]); // Унифицированное имя для списка
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<'lines' | 'info'>('lines');
  const [error, setError] = useState<string | null>(null);

  // FIX: prevent duplicate line creation race condition
  const isCreatingLineRef = useRef(false);

  // FIX: храним папку клиента, полученную при загрузке дела
  const [clientFolderName, setClientFolderName] = useState('');

  // ── Состояние модалок (согласно стандарту useRegistryLogic) ──
  const [selectedItem, setSelectedItem] = useState<Line | null>(null);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);

  const [formData, setFormData] = useState<Partial<Line>>({});
  const [isSaving, setIsSaving] = useState(false);

  // ── Состояние для контента (консультации/прочее) ──
  const [defaultLine, setDefaultLine] = useState<Line | null>(null);
  const [appeals, setAppeals] = useState<any[]>([]);

  // ── Загрузка данных ──
  const fetchItems = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = await businessApi.getCaseDetails(caseId);
      const caseItem = payload.case_item as Case;
      setCase(caseItem);
      const lines = payload.lines as Line[];
      setItems(lines);

      // Если это консультация, договор, прочее или судебное — берем первую линию как дефолтную или создаем её
      // Для судебных дел создаем линию в БД но не создаем папку на диске
      if (['consulting', 'other', 'contract', 'litigation'].includes(caseItem.case_type || '')) {
        let line = lines[0];
        if (!line) {
          // Автоматически создаем дефолтную линию
          line = { id: 0, title: t('common.loading' as any), case_id: caseId } as Line;
        }
        setDefaultLine(line);

        if (line && line.id !== 0) {
          const lineDetails = await businessApi.getLineDetails(line.id);
          setAppeals(lineDetails.appeals);
        }
      }

      // FIX: получаем folder_name клиента
      if (caseItem.client_id) {
        try {
          const clientPayload = await businessApi.getClientDetails(caseItem.client_id);
          setClientFolderName(clientPayload.client.folder_name ?? '');

          // Если линия была новой (id=0), теперь у нас есть данные для её создания
          // Для судебных дел создаем линию в БД но не создаем папку на диске
          if (
            ['consulting', 'other', 'contract', 'litigation'].includes(caseItem.case_type || '') &&
            lines.length === 0 &&
            !isCreatingLineRef.current
          ) {
            isCreatingLineRef.current = true;
            try {
              await businessApi.saveLine({
                case_id: caseId,
                case_folder_name: caseItem.folder_name,
                client_folder_name: clientPayload.client.folder_name ?? '',
                case_type: caseItem.case_type,
                title: t('line.defaultSectionTitle' as any),
                status: 'ACTIVE',
              } as any);
              // Перезагружаем чтобы получить честный ID
              const updatedPayload = await businessApi.getCaseDetails(caseId);
              const firstLine = updatedPayload.lines[0];
              setDefaultLine(firstLine);
              const lineDetails = await businessApi.getLineDetails(firstLine.id);
              setAppeals(lineDetails.appeals);
            } finally {
              isCreatingLineRef.current = false;
            }
          }
        } catch {
          setClientFolderName('');
        }
      }
    } catch (err) {
      console.error('[CaseDetails] fetch error:', err);
      setError((err as AppErrorLike).key ?? (err as AppErrorLike).message ?? 'case.error.load');
    } finally {
      setLoading(false);
    }
  }, [caseId]);

  const handleUpdateCase = async (updates: Partial<Case>) => {
    if (!case_) return;
    try {
      await businessApi.saveCase({
        ...case_,
        ...updates,
        client_folder_name: clientFolderName,
      } as any);
      void fetchItems();
    } catch (err) {
      console.error('Failed to update case:', err);
    }
  };

  const handleUpdateDefaultLine = async (updates: Partial<Line>) => {
    if (!defaultLine || !case_) return;
    try {
      await businessApi.saveLine({
        ...defaultLine,
        ...updates,
        case_id: caseId,
        case_folder_name: case_.folder_name,
        client_folder_name: clientFolderName,
        case_type: case_.case_type,
      } as any);
      void fetchItems();
    } catch (err) {
      console.error('Failed to update line:', err);
    }
  };

  const handleSaveAppeal = async (appealData: any) => {
    if (!defaultLine || !case_) return;
    try {
      await businessApi.saveAppeal({
        ...appealData,
        line_id: defaultLine.id,
        case_folder_name: case_.folder_name,
        client_folder_name: clientFolderName,
        line_folder_name: defaultLine.folder_name,
        case_type: case_.case_type,
      });
      // Refresh appeals from the line
      if (defaultLine.id !== 0) {
        const lineDetails = await businessApi.getLineDetails(defaultLine.id);
        setAppeals(lineDetails.appeals);
      } else {
        void fetchItems();
      }
    } catch (err) {
      console.error('Failed to save appeal:', err);
    }
  };

  const handleUploadDocument = async (file?: File) => {
    if (!defaultLine || defaultLine.id === 0 || !case_ || !clientFolderName) {
      console.warn('Cannot upload: default line not ready or data missing', {
        lineId: defaultLine?.id,
        caseReady: !!case_,
        clientReady: !!clientFolderName,
      });
      return;
    }
    try {
      let absolutePath = file ? (file as any).path : null;
      let fileName = file ? file.name : '';

      if (!absolutePath) {
        const { open } = await import('@tauri-apps/plugin-dialog');
        const selected = await open({
          multiple: false,
          filters: [
            { name: 'Document', extensions: ['pdf', 'docx', 'txt', 'jpg', 'png', 'xlsx', 'xls'] },
          ],
        });
        if (!selected || Array.isArray(selected)) return;
        absolutePath = selected;
        fileName = selected.split('/').pop()?.split('\\').pop() ?? 'document';
      }

      // 1. Стандартное имя для базы
      let textPreview = '';
      try {
        const ext = fileName.split('.').pop()?.toLowerCase();
        if (ext && ['txt', 'md', 'json', 'csv'].includes(ext)) {
          const { readTextFile } = await import('@tauri-apps/plugin-fs');
          textPreview = (await readTextFile(absolutePath as string)).slice(0, 300);
        }
      } catch {
        textPreview = '';
      }

      const classification = await DocumentClassifierAgent.classify(fileName, textPreview);
      const title =
        classification.confidence > 0.7 ? classification.label : fileName.replace(/\.[^/.]+$/, '');

      // 2. Создаем запись в БД (appeal)
      const appealId = await businessApi.saveAppeal({
        title,
        status: 'DRAFT',
        line_id: defaultLine.id,
        case_folder_name: case_.folder_name,
        client_folder_name: clientFolderName,
        line_folder_name: defaultLine.folder_name,
        case_type: case_.case_type,
      });

      // 3. Загружаем файл физически и регистрируем его
      await businessApi.uploadAppealDocument(appealId, absolutePath);
      if (classification.confidence > 0.7) {
        console.info(`[DocumentClassifier] Тип определён автоматически: ${classification.label}`);
      }

      void fetchItems();
    } catch (err) {
      console.error('[CaseDetails] Failed to upload document:', err);
      setError('errors.document.uploadFailed');
    }
  };

  const handleDeleteAppeal = async (appealId: number) => {
    try {
      await businessApi.deleteAppeal(appealId);
      void fetchItems();
    } catch (err) {
      console.error('Failed to delete appeal:', err);
    }
  };

  // FIX: добавлен void — Promise не теряется
  useEffect(() => {
    void fetchItems();
  }, [fetchItems]);

  // ── Управление модалками ──
  const openCreate = () => {
    setSelectedItem(null);
    setFormData({
      title: '',
      description: '',
    });
    setIsEditOpen(true);
  };

  const openEdit = (item: Line) => {
    setSelectedItem(item);
    setFormData({ ...item });
    setIsEditOpen(true);
  };

  const openDelete = (item: Line) => {
    setSelectedItem(item);
    setIsDeleteOpen(true);
  };

  // ── Сохранение ──
  const handleSave = async () => {
    if (!formData.title?.trim() || !case_) return;

    setIsSaving(true);
    try {
      const payload = {
        id: selectedItem?.id,
        case_id: caseId,
        case_folder_name: case_.folder_name,
        client_folder_name: clientFolderName,
        case_type: case_.case_type,
        title: formData.title.trim(),
        status: formData.status || 'ACTIVE',
        description: formData.description || '',
        consultation_text: formData.consultation_text,
        consultation_topic: formData.consultation_topic,
        contract_analysis: formData.contract_analysis,
        contract_risks: formData.contract_risks,
      };

      await businessApi.saveLine(payload as any);
      await fetchItems();
      setIsEditOpen(false);
    } catch (err) {
      console.error('[CaseDetails] save error:', err);
      setError((err as AppErrorLike).key ?? (err as AppErrorLike).message ?? 'case.error.save');
    } finally {
      setIsSaving(false);
    }
  };

  // ── Удаление ──
  const confirmDelete = async () => {
    if (!selectedItem) return;
    try {
      await businessApi.deleteLine(Number(selectedItem.id));
      await fetchItems();
      setIsDeleteOpen(false);
    } catch (err) {
      console.error('[CaseDetails] delete error:', err);
      setError((err as AppErrorLike).key ?? (err as AppErrorLike).message ?? 'case.error.delete');
    }
  };

  return {
    // Данные
    case_,
    lines: items,
    loading,
    error,
    view,
    setView,

    // Состояния
    selectedItem,
    isEditOpen,
    isDeleteOpen,
    setIsEditOpen,
    setIsDeleteOpen,

    // Форма
    formData,
    setFormData,
    isSaving,

    // Действия
    openCreate,
    openEdit,
    openDelete,
    handleSave,
    confirmDelete,
    fetchItems,
    closeEdit: () => setIsEditOpen(false),
    closeDelete: () => setIsDeleteOpen(false),
    // New props for consulting/other/litigation
    defaultLine,
    appeals,
    handleUpdateCase,
    handleUpdateDefaultLine,
    handleSaveAppeal,
    handleUploadDocument,
    handleDeleteAppeal,
  };
}
