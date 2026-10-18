import { Info, Plus, FileCheck, FileText, User, Hash, Calendar, Save, Sparkles, Loader2, FileBox, ShieldCheck } from 'lucide-react';
import { PageHeader } from '../../../components/ui/layout/PageHeader';
import { Window } from '../../../components/ui/layout/Window';
import { SegmentedControl } from '../../../components/ui/layout/SegmentedControl';
import { PageLoader } from '../../../components/ui/feedback/PageLoader';
import { Button } from '../../../components/ui/form/Button';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { AppealsTable } from '../../LineDetails/components/AppealsTable';
import { CaseInfoPanel } from '../components/CaseInfoPanel';
import { Case, Line, Appeal } from '../../../types/entities';
import { Textarea } from '../../../components/ui/form/Textarea';
import { Input } from '../../../components/ui/form/Input';
import { Select } from '../../../components/ui/form/Select';
import { useState, useEffect, useMemo } from 'react';
import { RegistryToolbar } from '../../../shared/registry/RegistryToolbar';
import { AppealEditModal } from '../../LineDetails/components/AppealEditModal';
import { FileDropZone } from '../../../shared/components/FileDropZone';
import { useAnalysis } from '../../laila/hooks/useAnalysis';
import { businessApi } from '../../../services/backend/businessApi';
import { ShellService } from '../../../services/shell/ShellService';

interface ContractScenarioProps {
  case_: Case | null;
  loading: boolean;
  view: 'lines' | 'info';
  setView: (view: 'lines' | 'info') => void;
  onBack: () => void;
  defaultLine: Line | null;
  appeals: Appeal[];
  handleUpdateCase: (updates: Partial<Case>) => Promise<void>;
  handleUpdateDefaultLine: (updates: Partial<Line>) => Promise<void>;
  handleSaveAppeal: (data: any) => Promise<void>;
  handleUploadDocument: (file?: File) => Promise<void>;
  handleDeleteAppeal: (id: number) => Promise<void>;
}

export function ContractScenario({
  case_,
  loading,
  view,
  setView,
  onBack,
  defaultLine,
  appeals,
  handleUpdateCase,
  handleUpdateDefaultLine,
  handleSaveAppeal,
  handleUploadDocument,
  handleDeleteAppeal,
}: ContractScenarioProps) {
  const { t } = useI18n();
  const [localAnalysis, setLocalAnalysis] = useState(defaultLine?.contract_analysis || '');
  const [localTitle, setLocalTitle] = useState(case_?.title || '');
  const [localCounterparty, setLocalCounterparty] = useState(case_?.counterparty || '');
  const [localNumber, setLocalNumber] = useState(case_?.case_number || '');
  const [localDate, setLocalDate] = useState(case_?.contract_date || '');
  const [localStatus, setLocalStatus] = useState(case_?.status || 'ACTIVE');

  const [isAppealModalOpen, setIsAppealModalOpen] = useState(false);
  const [editingAppeal, setEditingAppeal] = useState<Appeal | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isSavingCase, setIsSavingCase] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortType, setSortType] = useState<'date_desc' | 'date_asc' | 'name_asc' | 'name_desc'>('date_desc');
  const [statusFilter, setStatusFilter] = useState<'all' | 'active' | 'done' | 'archive'>('all');

  const filteredAppeals = useMemo(() => {
    let result = appeals;

    // Search filter
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      result = result.filter(
        (a) => a.title.toLowerCase().includes(q) || a.agency_code?.toLowerCase().includes(q),
      );
    }

    // Status filter
    if (statusFilter !== 'all') {
      result = result.filter((a) => a.status?.toLowerCase() === statusFilter);
    }

    // Sort
    result = [...result].sort((a, b) => {
      switch (sortType) {
        case 'date_desc':
          return new Date(b.appeal_date || 0).getTime() - new Date(a.appeal_date || 0).getTime();
        case 'date_asc':
          return new Date(a.appeal_date || 0).getTime() - new Date(b.appeal_date || 0).getTime();
        case 'name_asc':
          return a.title.localeCompare(b.title);
        case 'name_desc':
          return b.title.localeCompare(a.title);
        default:
          return 0;
      }
    });

    return result;
  }, [appeals, searchQuery, statusFilter, sortType]);

  const laila = useAnalysis('case-contract');

  const statusOptions = useMemo(
    () => [
      { value: 'LEAD', label: t('case.status.LEAD' as any) },
      { value: 'ACTIVE', label: t('case.status.ACTIVE' as any) },
      { value: 'PAUSED', label: t('cases.status.PAUSED' as any) },
      { value: 'ARCHIVE', label: t('case.status.ARCHIVE' as any) },
    ],
    [t],
  );

  const hasChanges =
    localTitle !== (case_?.title || '') ||
    localCounterparty !== (case_?.counterparty || '') ||
    localNumber !== (case_?.case_number || '') ||
    localDate !== (case_?.contract_date || '') ||
    localStatus !== (case_?.status || 'ACTIVE');

  useEffect(() => {
    if (defaultLine) {
      setLocalAnalysis(defaultLine.contract_analysis || '');
    }
  }, [defaultLine]);

  useEffect(() => {
    if (case_) {
      setLocalTitle(case_.title || '');
      setLocalCounterparty(case_.counterparty || '');
      setLocalNumber(case_.case_number || '');
      setLocalDate(case_.contract_date || '');
      setLocalStatus(case_.status || 'ACTIVE');
    }
  }, [case_]);

  const onSaveAnalysis = () => {
    handleUpdateDefaultLine({ contract_analysis: localAnalysis });
  };

  const onSaveCase = async () => {
    setIsSavingCase(true);
    try {
      await handleUpdateCase({
        title: localTitle,
        counterparty: localCounterparty,
        case_number: localNumber,
        contract_date: localDate,
        status: localStatus,
      });
    } finally {
      setIsSavingCase(false);
    }
  };

  const handleOpenDocument = async (appeal: Appeal) => {
    try {
      const payload = await businessApi.getAppealWorkbench(appeal.id);
      const file = payload.files[0];
      if (file?.path) {
        await ShellService.open(file.path);
        return;
      }
      if (payload.details.folder_path) {
        await ShellService.open(payload.details.folder_path);
      }
    } catch (err) {
      console.error('[ContractScenario] failed to open document', err);
    }
  };

  const onRunLaila = async () => {
    laila.setAnalysisText(localAnalysis);
    await laila.runAnalysis();
  };

  useEffect(() => {
    if (laila.analysisResult?.summary) {
      setLocalAnalysis((prev) => `${prev}\n\n--- ${t('caseDetails.contract.aiAnalysisDivider' as any)} ---\n${laila.analysisResult?.summary}`);
    }
  }, [laila.analysisResult, t]);

  const handleFilesDrop = async (files: File[]) => {
    setIsUploading(true);
    try {
      for (const file of files) {
        await handleUploadDocument(file);
      }
    } finally {
      setIsUploading(false);
    }
  };

  if (loading || !case_ || !defaultLine) return <PageLoader />;

  return (
    <div className="flex flex-col h-full bg-[var(--app-bg)] overflow-hidden animate-in fade-in duration-500">
      <PageHeader
        title={localTitle || t('caseDetails.contract.defaultTitle' as any)}
        subtitle={case_?.folder_name || ''}
        icon={<FileCheck size={20} className="text-[var(--app-accent)]" />}
        onBack={onBack}
        rightContent={
          <div className="flex items-center gap-3">
            <Button
              variant="primary"
              size="sm"
              icon={isSavingCase ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
              onClick={onSaveCase}
              loading={isSavingCase}
              disabled={!hasChanges}
            >
              {t('case.modal.save')}
            </Button>
            <SegmentedControl
              active={view}
              onChange={(id) => setView(id as 'lines' | 'info')}
              segments={[
                { id: 'lines', label: t('caseDetails.contract.tabLabel' as any), icon: <FileText size={14} /> },
                { id: 'info', label: t('appeal.info'), icon: <Info size={14} /> },
              ]}
            />
          </div>
        }
      />

      <div className="flex-1 overflow-hidden p-4">
        {view === 'info' ? (
          <Window noPadding className="h-full overflow-auto mica-effect">
            <CaseInfoPanel case_={case_} />
          </Window>
        ) : (
          <div className="h-full flex flex-col gap-4 overflow-hidden">
            <div className="flex gap-4 shrink-0">
              <Window noPadding className="flex-1 mica-effect py-3 px-6 bg-white/40">
                <div className="flex items-center gap-6">
                  <div className="flex flex-col gap-1 min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 opacity-40">
                      <FileText size={11} />
                      <span className="text-[10px] font-black uppercase tracking-widest">{t('caseDetails.contract.subjectLabel')}</span>
                    </div>
                    <Input
                      value={localTitle}
                      onChange={(e) => setLocalTitle(e.target.value)}
                      placeholder={t('caseDetails.contract.titlePlaceholder')}
                      className="font-bold text-[14px] p-0 h-auto bg-transparent border-none shadow-none focus:ring-0"
                    />
                  </div>

                  <div className="w-px h-8 bg-black/[0.05]" />

                  <div className="flex flex-col gap-1 min-w-0 flex-[0.8]">
                    <div className="flex items-center gap-1.5 opacity-40">
                      <User size={11} />
                      <span className="text-[10px] font-black uppercase tracking-widest">{t('caseDetails.contract.counterpartyLabel')}</span>
                    </div>
                    <Input
                      value={localCounterparty}
                      onChange={(e) => setLocalCounterparty(e.target.value)}
                      placeholder={t('caseDetails.contract.counterpartyPlaceholder')}
                      className="font-semibold text-[13px] p-0 h-auto bg-transparent border-none shadow-none focus:ring-0"
                    />
                  </div>

                  <div className="w-px h-8 bg-black/[0.05]" />

                  <div className="flex flex-col gap-1 w-28">
                    <div className="flex items-center gap-1.5 opacity-40">
                      <Hash size={11} />
                      <span className="text-[10px] font-black uppercase tracking-widest">{t('caseDetails.contract.numberLabel')}</span>
                    </div>
                    <Input
                      value={localNumber}
                      onChange={(e) => setLocalNumber(e.target.value)}
                      placeholder={t('caseDetails.contract.numberPlaceholder')}
                      className="font-mono text-[12px] p-0 h-auto bg-transparent border-none shadow-none focus:ring-0"
                    />
                  </div>

                  <div className="w-px h-8 bg-black/[0.05]" />

                  <div className="flex flex-col gap-1 w-32">
                    <div className="flex items-center gap-1.5 opacity-40">
                      <Calendar size={11} />
                      <span className="text-[10px] font-black uppercase tracking-widest">{t('caseDetails.contract.dateLabel')}</span>
                    </div>
                    <Input
                      type="date"
                      value={localDate}
                      onChange={(e) => setLocalDate(e.target.value)}
                      className="text-[12px] p-0 h-auto bg-transparent border-none shadow-none focus:ring-0"
                    />
                  </div>

                  <div className="w-px h-8 bg-black/[0.05]" />

                  <div className="flex flex-col gap-1 w-36">
                    <div className="flex items-center gap-1.5 opacity-40">
                      <ShieldCheck size={11} />
                      <span className="text-[10px] font-black uppercase tracking-widest">{t('caseDetails.contract.statusLabel')}</span>
                    </div>
                    <Select
                      value={localStatus}
                      onChange={setLocalStatus}
                      options={statusOptions}
                      className="text-[12px] p-0 h-auto bg-transparent border-none shadow-none focus:ring-0 font-bold"
                    />
                  </div>
                </div>
              </Window>
            </div>

            <div className="flex-1 flex gap-4 overflow-hidden">
              <div className="flex-[3] flex flex-col min-w-0">
                <Window
                  noPadding
                  className="flex-1 flex flex-col overflow-hidden mica-effect"
                  title={t('caseDetails.contract.analysisTitle')}
                >
                  <div className="p-3 border-b border-black/[0.05] bg-[var(--app-surface-soft)] flex items-center justify-between">
                    <span className="text-[10px] italic opacity-60 text-[var(--app-text-secondary)]">
                      {t('caseDetails.contract.analysisHint')}
                    </span>
                    <Button
                      variant="secondary"
                      size="sm"
                      icon={laila.isAnalyzing ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                      onClick={onRunLaila}
                      disabled={laila.isAnalyzing || !localAnalysis.trim()}
                    >
                      {laila.isAnalyzing ? t('caseDetails.contract.analyzing') : t('caseDetails.contract.analyzeAction')}
                    </Button>
                  </div>
                  <div className="flex-1 p-6 flex flex-col gap-4 bg-white/40">
                    <Textarea
                      value={laila.isAnalyzing ? laila.streamingText : localAnalysis}
                      onChange={(e) => setLocalAnalysis(e.target.value)}
                      onBlur={() => onSaveAnalysis()}
                      placeholder={t('caseDetails.contract.analysisPlaceholder')}
                      className="flex-1 resize-none bg-white/60 shadow-inner border-black/[0.05] focus:border-[var(--app-accent)] focus:ring-[var(--app-accent)]/20 text-[14px] leading-relaxed"
                    />
                  </div>
                </Window>
              </div>

              <div className="flex-[2] flex flex-col min-w-[420px]">
                <Window noPadding className="flex-1 flex flex-col overflow-hidden mica-effect">
                  <FileDropZone
                    className="flex-1 flex flex-col overflow-hidden"
                    onFilesDrop={handleFilesDrop}
                    isUploading={isUploading}
                  >
                    <div className="flex flex-col h-full bg-white/40 overflow-hidden">
                      <RegistryToolbar
                        searchPlaceholder={t('caseDetails.contract.searchPlaceholder')}
                        searchValue={searchQuery}
                        onSearchChange={setSearchQuery}
                        filters={
                          <>
                            <Select
                              value={sortType}
                              onChange={(v) => setSortType(v as typeof sortType)}
                              options={[
                                { label: t('clients.sort.newest'), value: 'date_desc' },
                                { label: t('clients.sort.oldest'), value: 'date_asc' },
                                { label: t('clients.sort.nameAsc'), value: 'name_asc' },
                                { label: t('clients.sort.nameDesc'), value: 'name_desc' },
                              ]}
                            />
                            <Select
                              value={statusFilter}
                              onChange={(v) => setStatusFilter(v as typeof statusFilter)}
                              options={[
                                { label: t('clients.filter.status.all'), value: 'all' },
                                { label: t('clients.filter.status.active'), value: 'active' },
                                { label: t('clients.filter.status.done'), value: 'done' },
                                { label: t('clients.filter.status.archive'), value: 'archive' },
                              ]}
                            />
                          </>
                        }
                        rightContent={
                          <Button
                            variant="primary"
                            size="sm"
                            icon={<Plus size={14} />}
                            onClick={() => handleUploadDocument()}
                          >
                            {t('caseDetails.contract.addVersion')}
                          </Button>
                        }
                      />
                      <div className="flex-1 overflow-auto">
                        <AppealsTable
                          items={filteredAppeals}
                          onEdit={(a) => {
                            setEditingAppeal(a);
                            setIsAppealModalOpen(true);
                          }}
                          onDelete={(a) => handleDeleteAppeal(a.id)}
                          onNavigate={handleOpenDocument}
                          compact
                          hideHeader
                        />
                        {filteredAppeals.length === 0 && (
                          <div className="py-20 text-center flex flex-col items-center gap-3 opacity-30">
                            <div className="w-16 h-16 rounded-full bg-black/[0.03] flex items-center justify-center">
                              <FileBox size={32} strokeWidth={1} />
                            </div>
                            <div className="flex flex-col gap-1">
                              <span className="text-[10px] font-bold">{t('caseDetails.contract.documentsEmptyTitle')}</span>
                              <span className="text-[10px] text-[var(--app-text-secondary)]">{t('caseDetails.contract.documentsEmptyHint')}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </FileDropZone>
                </Window>
              </div>
            </div>
          </div>
        )}
      </div>

      {isAppealModalOpen && (
        <AppealEditModal
          open={isAppealModalOpen}
          onClose={() => setIsAppealModalOpen(false)}
          onSave={async () => {
            await handleSaveAppeal(editingAppeal || {});
            setIsAppealModalOpen(false);
          }}
          formData={editingAppeal || {}}
          setFormData={(data) => setEditingAppeal(data as Appeal)}
          isNew={!editingAppeal}
          isSaving={false}
          titleOverride={t('caseDetails.contract.documentModalTitle' as any)}
        />
      )}
    </div>
  );
}
