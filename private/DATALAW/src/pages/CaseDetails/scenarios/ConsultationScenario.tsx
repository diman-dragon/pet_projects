import { HelpCircle, Info, Plus, MessageSquare, Loader2, Save, Sparkles, FileBox } from 'lucide-react';
import { PageHeader } from '../../../components/ui/layout/PageHeader';
import { Window } from '../../../components/ui/layout/Window';
import { SegmentedControl } from '../../../components/ui/layout/SegmentedControl';
import { PageLoader } from '../../../components/ui/feedback/PageLoader';
import { AppButton } from '../../../components/ui/form/AppButton';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { AppealsTable } from '../../LineDetails/components/AppealsTable';
import { CaseInfoPanel } from '../components/CaseInfoPanel';
import { Case, Line, Appeal } from '../../../types/entities';
import { Textarea } from '../../../components/ui/form/Textarea';
import { useState, useEffect, useMemo } from 'react';
import { RegistryToolbar } from '../../../shared/registry/RegistryToolbar';
import { Select } from '../../../components/ui/form/Select';
import { AppealEditModal } from '../../LineDetails/components/AppealEditModal';
import { useAnalysis } from '../../laila/hooks/useAnalysis';
import { FileDropZone } from '../../../shared/components/FileDropZone';
import { businessApi } from '../../../services/backend/businessApi';
import { ShellService } from '../../../services/shell/ShellService';

interface ConsultationScenarioProps {
  case_: Case | null;
  loading: boolean;
  view: 'lines' | 'info';
  setView: (view: 'lines' | 'info') => void;
  onBack: () => void;
  defaultLine: Line | null;
  appeals: Appeal[];
  handleUpdateDefaultLine: (updates: Partial<Line>) => Promise<void>;
  handleSaveAppeal: (data: any) => Promise<void>;
  handleUploadDocument: (file?: File) => Promise<void>;
  handleDeleteAppeal: (id: number) => Promise<void>;
}

export function ConsultationScenario({
  case_,
  loading,
  view,
  setView,
  onBack,
  defaultLine,
  appeals,
  handleUpdateDefaultLine,
  handleSaveAppeal,
  handleUploadDocument,
  handleDeleteAppeal,
}: ConsultationScenarioProps) {
  const { t } = useI18n();
  const [localText, setLocalText] = useState(defaultLine?.consultation_text || '');
  const [isAppealModalOpen, setIsAppealModalOpen] = useState(false);
  const [editingAppeal, setEditingAppeal] = useState<Appeal | null>(null);
  const [isUploading, setIsUploading] = useState(false);
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
  const [isSaving, setIsSaving] = useState(false);

  const laila = useAnalysis('case-consultation');

  const hasChanges = localText !== (defaultLine?.consultation_text || '');

  useEffect(() => {
    if (defaultLine) {
      setLocalText(defaultLine.consultation_text || '');
    }
  }, [defaultLine]);

  const onSaveText = async () => {
    setIsSaving(true);
    try {
      await handleUpdateDefaultLine({ consultation_text: localText });
    } finally {
      setIsSaving(false);
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
      console.error('[ConsultationScenario] failed to open document', err);
    }
  };

  const onRunLaila = async () => {
    laila.setAnalysisText(localText);
    await laila.runAnalysis();
  };

  useEffect(() => {
    if (laila.analysisResult?.summary) {
      setLocalText((prev) => `${prev}\n\n--- ${t('caseDetails.consultation.aiAnalysisDivider')} ---\n${laila.analysisResult?.summary}`);
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

  return (
    <div className="flex flex-col h-full bg-[var(--app-bg)] overflow-hidden animate-in fade-in duration-300">
      <PageHeader
        title={t('caseDetails.consultation.pageTitle')}
        subtitle={loading ? t('loader.loading') : case_?.title || ''}
        icon={<HelpCircle size={20} className="text-[var(--app-accent)]" />}
        onBack={onBack}
        rightContent={
          <div className="flex items-center gap-3">
            <AppButton
              variant="primary"
              size="sm"
              icon={<Save size={14} />}
              onClick={onSaveText}
              loading={isSaving}
              disabled={!hasChanges}
            >
              {t('appeal.modal.save')}
            </AppButton>
            <SegmentedControl
              active={view}
              onChange={(id) => setView(id as 'lines' | 'info')}
              segments={[
                { id: 'lines', label: t('caseDetails.consultation.contentTab'), icon: <MessageSquare size={14} /> },
                { id: 'info', label: t('appeal.info'), icon: <Info size={14} /> },
              ]}
            />
          </div>
        }
      />

      <div className="flex-1 overflow-hidden p-4">
        {loading || !defaultLine ? (
          <PageLoader text={t('loader.loading')} />
        ) : view === 'info' ? (
          <Window noPadding className="h-full overflow-auto mica-effect">
            <CaseInfoPanel case_={case_} />
          </Window>
        ) : (
          <div className="h-full flex gap-4 overflow-hidden">
            <div className="flex-[3] flex flex-col min-w-0">
              <Window
                noPadding
                className="flex-1 flex flex-col overflow-hidden mica-effect"
                title={t('caseDetails.consultation.textTitle')}
              >
                <div className="p-3 border-b border-black/[0.05] bg-[var(--app-surface-soft)] flex items-center gap-2">
                  <AppButton
                    variant="secondary"
                    size="sm"
                    icon={laila.isAnalyzing ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                    onClick={onRunLaila}
                    disabled={laila.isAnalyzing || !localText.trim()}
                  >
                    {laila.isAnalyzing ? t('caseDetails.consultation.analyzing') : t('caseDetails.consultation.analyzeAction')}
                  </AppButton>
                </div>
                <div className="flex-1 p-6 flex flex-col gap-4 bg-white/40">
                  <Textarea
                    value={laila.isAnalyzing ? laila.streamingText : localText}
                    onChange={(e) => setLocalText(e.target.value)}
                    onBlur={() => onSaveText()}
                    placeholder={t('caseDetails.consultation.inputPlaceholder')}
                    className="flex-1 resize-none bg-white/60 shadow-inner border-black/[0.05] focus:border-[var(--app-accent)] focus:ring-[var(--app-accent)]/20 text-[14px] leading-relaxed"
                  />
                </div>
              </Window>
            </div>

            <div className="flex-[2] flex flex-col min-w-[450px]">
              <Window noPadding className="flex-1 flex flex-col overflow-hidden mica-effect">
                <FileDropZone
                  className="flex-1 flex flex-col overflow-hidden"
                  onFilesDrop={handleFilesDrop}
                  isUploading={isUploading}
                >
                  <div className="flex flex-col h-full bg-white/40 overflow-hidden">
                    <RegistryToolbar
                      searchPlaceholder={t('caseDetails.consultation.searchPlaceholder')}
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
                        <AppButton
                          variant="primary"
                          size="sm"
                          icon={<Plus size={14} />}
                          onClick={() => handleUploadDocument()}
                        >
                          {t('caseDetails.consultation.pickFile')}
                        </AppButton>
                      }
                    />
                    <div className="flex-1 overflow-auto min-h-0">
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
                        <div className="py-20 text-center flex flex-col items-center gap-2 opacity-30">
                          <FileBox size={32} strokeWidth={1} />
                          <span className="text-[10px] font-medium">{t('caseDetails.consultation.materialsEmpty')}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </FileDropZone>
              </Window>
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
        />
      )}
    </div>
  );
}
