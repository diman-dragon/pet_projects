// src/pages/LineDetails/scenarios/LitigationLineScenario.tsx
import { ListTree, FileText, Info, Plus } from 'lucide-react';
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

interface LitigationLineScenarioProps {
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

export function LitigationLineScenario({
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
}: LitigationLineScenarioProps) {
  const { t } = useI18n();

  return (
    <div className="flex flex-col h-full bg-white overflow-hidden animate-in fade-in duration-300">
      <PageHeader
        title={t('line.pageTitle')}
        subtitle={loading ? t('loader.loading') : line.title}
        icon={<ListTree size={20} className="text-[var(--app-accent)]" />}
        onBack={onBack}
        rightContent={
          <div className="flex items-center gap-3">
            <SegmentedControl
              active={view}
              onChange={(id) => setView(id as 'appeals' | 'info')}
              segments={[
                { id: 'appeals', label: t('appeal.type'), icon: <FileText size={14} /> },
                { id: 'info', label: t('appeal.info'), icon: <Info size={14} /> },
              ]}
            />
            <Button
              variant="primary"
              size="md"
              onClick={openCreate}
              icon={<Plus size={16} />}
            >
              {t('line.newAppeal')}
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
          <div className="flex flex-col h-full overflow-y-auto">
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
      />

      <DeleteConfirmModal
        target={deleteTarget}
        entityLabel={t('appeal.type').toLowerCase()}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={handleDeleteAppeal}
      />
    </div>
  );
}
