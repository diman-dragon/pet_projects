// src/pages/LineDetails/scenarios/OtherLineScenario.tsx
import { FileText, Info, Plus, Folder } from 'lucide-react';
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

interface OtherLineScenarioProps {
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
}

export function OtherLineScenario({
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
}: OtherLineScenarioProps) {
  const { t } = useI18n();

  return (
    <div className="flex flex-col h-full bg-slate-50 overflow-hidden animate-in fade-in duration-300">
      <PageHeader
        title={t('line.scenario.other.title')}
        subtitle={loading ? t('loader.loading') : line.title}
        icon={<Folder size={20} className="text-slate-600" />}
        onBack={onBack}
        rightContent={
          <div className="flex items-center gap-3">
            <SegmentedControl
              active={view}
              onChange={(id) => setView(id as 'appeals' | 'info')}
              segments={[
                { id: 'appeals', label: t('line.scenario.other.appealsTab'), icon: <FileText size={14} /> },
                { id: 'info', label: t('appeal.info'), icon: <Info size={14} /> },
              ]}
            />
            <Button
              variant="primary"
              size="md"
              onClick={openCreate}
              icon={<Plus size={16} />}
              className="bg-slate-600 hover:bg-slate-700"
            >
              Добавить документ
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
          <div className="flex flex-col h-full overflow-y-auto bg-white">
            <AppealsTable
              items={appeals}
              onEdit={openEdit}
              onDelete={setDeleteTarget}
              onNavigate={onNavigateAppeal}
            />
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
        subtitleOverride="Файлы и материалы производства"
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
