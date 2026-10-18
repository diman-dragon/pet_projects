// src/pages/LineDetails/scenarios/ContractSectionScenario.tsx
import { FileText, Info, Plus, LayoutGrid, FileCheck, Sparkles, Save, ShieldAlert } from 'lucide-react';
import { PageHeader } from '../../../components/ui/layout/PageHeader';
import { Window } from '../../../components/ui/layout/Window';
import { SegmentedControl } from '../../../components/ui/layout/SegmentedControl';
import { PageLoader } from '../../../components/ui/feedback/PageLoader';
import { Button } from '../../../components/ui/form/Button';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { AppealsTable } from '../components/AppealsTable';
import { InfoTab } from '../components/InfoTab';
import { AppealEditModal } from '../components/AppealEditModal';
import { DeleteConfirmModal } from '../../../components/ui/overlays/DeleteConfirmModal';
import { Appeal, Line } from '../../../types/entities';
import { Textarea } from '../../../components/ui/form/Textarea';
import { useState, useEffect } from 'react';

interface ContractSectionScenarioProps {
  line: Line;
  appeals: Appeal[];
  loading: boolean;
  view: 'appeals' | 'info';
  setView: (view: 'appeals' | 'info') => void;
  onBack: () => void;
  onNavigateAppeal: (appeal: Appeal) => void;
  // Modals logic
  isNew: boolean;
  editingAppeal: Appeal | null;
  isSaving: boolean;
  formData: any;
  setFormData: (data: any) => void;
  openCreate: () => void;
  openEdit: (appeal: Appeal) => void;
  closeModal: () => void;
  handleSaveAppeal: () => void;
  deleteTarget: Appeal | null;
  setDeleteTarget: (target: Appeal | null) => void;
  handleDeleteAppeal: () => void;
  handleUpdateLine: (updates: Partial<Line>) => Promise<void>;
}

export function ContractSectionScenario({
  line,
  appeals,
  loading,
  view,
  setView,
  onBack,
  onNavigateAppeal,
  isNew,
  editingAppeal,
  isSaving,
  formData,
  setFormData,
  openCreate,
  openEdit,
  closeModal,
  handleSaveAppeal,
  deleteTarget,
  setDeleteTarget,
  handleDeleteAppeal,
  handleUpdateLine,
}: ContractSectionScenarioProps) {
  const { t } = useI18n();
  const [localAnalysis, setLocalAnalysis] = useState(line.contract_analysis || '');
  const [localRisks, setLocalRisks] = useState(line.contract_risks || '');

  useEffect(() => {
    setLocalAnalysis(line.contract_analysis || '');
    setLocalRisks(line.contract_risks || '');
  }, [line]);

  const handleSaveContractInfo = () => {
    handleUpdateLine({
      contract_analysis: localAnalysis,
      contract_risks: localRisks,
    });
  };

  return (
    <div className="flex flex-col h-full bg-slate-50 overflow-hidden animate-in fade-in duration-300">
      <PageHeader
        title={t('scenario.contract.sections')}
        subtitle={loading ? t('loader.loading') : line.title}
        icon={<LayoutGrid size={20} className="text-cyan-600" />}
        onBack={onBack}
        rightContent={
          <div className="flex items-center gap-3">
            <SegmentedControl
              active={view}
              onChange={(id) => setView(id as 'appeals' | 'info')}
              segments={[
                { id: 'appeals', label: "Анализ", icon: <FileCheck size={14} /> },
                { id: 'info', label: t('appeal.info'), icon: <Info size={14} /> },
              ]}
            />
            <Button
              variant="primary"
              size="md"
              onClick={openCreate}
              icon={<Plus size={16} />}
              className="bg-cyan-600 hover:bg-cyan-700"
            >
              {t('scenario.contract.addDocument')}
            </Button>
          </div>
        }
      />

      <Window className="flex-1 overflow-hidden" noPadding>
        {loading ? (
          <PageLoader text={t('loader.loading')} />
        ) : view === 'info' ? (
          <InfoTab line={line} appeals={appeals} />
        ) : (
          <div className="flex flex-col h-full overflow-hidden bg-white">
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              <div className="grid grid-cols-12 gap-6">
                <div className="col-span-8 space-y-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-[length:var(--text-sm)] font-semibold text-gray-900 flex items-center gap-2">
                        <FileText size={16} className="text-cyan-600" />
                        Краткий анализ
                      </h3>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          icon={<Sparkles size={14} />}
                          className="text-cyan-600 hover:bg-cyan-50"
                        >
                          Laila Анализ
                        </Button>
                        <Button
                          variant="primary"
                          size="sm"
                          onClick={handleSaveContractInfo}
                          loading={isSaving}
                          icon={<Save size={14} />}
                          className="bg-cyan-600"
                        >
                          Сохранить
                        </Button>
                      </div>
                    </div>
                    <Textarea
                      className="min-h-[200px] text-[length:var(--text-sm)] leading-relaxed border-cyan-100 focus:border-cyan-300 focus:ring-cyan-200"
                      placeholder="Общее описание раздела, ключевые условия..."
                      value={localAnalysis}
                      onChange={(e) => setLocalAnalysis(e.target.value)}
                    />
                  </div>

                  <div className="space-y-4">
                    <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                      <ShieldAlert size={16} className="text-red-500" />
                      Риски и замечания
                    </h3>
                    <Textarea
                      className="min-h-[150px] text-[length:var(--text-sm)] leading-relaxed border-red-100 focus:border-red-300 focus:ring-red-200 bg-red-50/30"
                      placeholder="Критические риски, спорные моменты..."
                      value={localRisks}
                      onChange={(e) => setLocalRisks(e.target.value)}
                    />
                  </div>
                </div>

                <div className="col-span-4 space-y-6">
                  <div className="space-y-3">
                    <h4 className="text-[length:var(--text-xs)] font-bold text-gray-500 uppercase tracking-wider px-1">Файлы раздела</h4>
                    <div className="rounded-xl border border-gray-100 overflow-hidden bg-gray-50/50">
                      <AppealsTable
                        items={appeals}
                        onEdit={openEdit}
                        onDelete={setDeleteTarget}
                        onNavigate={onNavigateAppeal}
                        compact
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </Window>

      <AppealEditModal
        open={isNew || !!editingAppeal}
        isNew={isNew}
        isSaving={isSaving}
        formData={formData}
        setFormData={setFormData}
        onClose={closeModal}
        onSave={handleSaveAppeal}
        titleOverride={isNew ? "Новый документ" : "Редактировать документ"}
        subtitleOverride="Файлы и материалы раздела договора"
      />

      <DeleteConfirmModal
        target={deleteTarget}
        entityLabel="документ"
        onCancel={() => setDeleteTarget(null)}
        onConfirm={handleDeleteAppeal}
      />
    </div>
  );
}
