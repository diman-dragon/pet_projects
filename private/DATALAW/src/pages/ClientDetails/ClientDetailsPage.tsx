// src/pages/ClientDetails/ClientDetailsPage.tsx

import { Briefcase, LayoutGrid, Info, Plus } from 'lucide-react';

import {
  PageHeader,
  Window,
  Button,
  Badge,
  DeleteConfirmModal,
  PageLoader
} from '@/components/ui';
import { SegmentedControl } from '@/components/ui/layout/SegmentedControl';
import { useI18n } from '@/i18n/AppLanguageProvider';

import { useClientDetailsLogic } from './useClientDetailsLogic';
import { CasesTab } from './components/CasesTab';
import { InfoTab } from './components/InfoTab';
import { CaseEditModal } from './components/CaseEditModal';
import CaseDetailsPage from '@/pages/CaseDetails/CaseDetailsPage';

type NavigationID = string | number | null;

interface ClientDetailsPageProps {
  clientId: string | number;
  onBack: () => void;
  activeCaseId: NavigationID;
  setActiveCaseId: (id: NavigationID) => void;
  activeLineId: NavigationID;
  setActiveLineId: (id: NavigationID) => void;
  activeAppealId: NavigationID;
  setActiveAppealId: (id: NavigationID) => void;
}

export default function ClientDetailsPage({
  clientId,
  onBack,
  activeCaseId,
  setActiveCaseId,
  activeLineId,
  setActiveLineId,
  activeAppealId,
  setActiveAppealId,
}: ClientDetailsPageProps) {
  // Получаем функцию перевода t
  const { t } = useI18n();

  const logic = useClientDetailsLogic(clientId);
  const numericClientId = Number(clientId);

  // Логика перехода в детали конкретного проекта (дела)
  if (activeCaseId) {
    return (
      <CaseDetailsPage
        caseId={activeCaseId}
        onBack={() => {
          setActiveCaseId(null);
          setActiveLineId(null);
          setActiveAppealId(null);
        }}
        activeLineId={activeLineId}
        setActiveLineId={setActiveLineId}
        activeAppealId={activeAppealId}
        setActiveAppealId={setActiveAppealId}
      />
    );
  }

  return (
    <div className="flex flex-col h-full bg-[var(--app-bg)] overflow-hidden">
      <PageHeader
        title={logic.client?.full_name || t('loader.loading')}
        subtitle={
          <div className="flex items-center gap-2 mt-0.5">
            <Badge
              label={logic.client?.type === 'LEGAL' ? 'LEGAL' : 'PRIV'}
              variant={logic.client?.type === 'LEGAL' ? 'blue' : 'amber'}
              className="text-[9px] px-1.5 py-0"
            />
            <span className="text-[11px] font-medium text-[var(--app-text-muted)]">
              {logic.client?.type === 'LEGAL'
                ? t('client.type.LEGAL' as any)
                : t('client.type.INDIVIDUAL' as any)}
            </span>
          </div>
        }
        icon={<Briefcase size={20} className="text-[var(--app-accent)]" />}
        onBack={onBack}
        rightContent={
          <div className="flex items-center gap-3">
            <SegmentedControl
              active={logic.view}
              onChange={(id) => logic.setView(id as 'cases' | 'info')}
              segments={[
                {
                  id: 'cases',
                  label: t('globalSearch.section.cases'),
                  icon: <LayoutGrid size={14} />,
                },
                {
                  id: 'info',
                  label: t('appeal.info'),
                  icon: <Info size={14} />,
                },
              ]}
            />

            <Button
              variant="primary"
              size="md"
              onClick={logic.openCreate}
              icon={<Plus size={16} />}
            >
              {t('client.newCase' as any)}
            </Button>
          </div>
        }
      />

      <Window className="flex-1 overflow-hidden" noPadding>
        {logic.loading ? (
          <PageLoader text={t('loader.loading')} />
        ) : logic.view === 'info' && logic.client ? (
          <div className="h-full overflow-y-auto">
            <InfoTab
              client={logic.client}
              cases={logic.cases}
              clientId={numericClientId}
              isSummaryGenerating={logic.isSummaryGenerating}
              onGenerateSummary={logic.generateSummary}
              isRiskScanning={logic.isRiskScanning}
              criticalRisksCount={logic.criticalRisksCount}
              onScanRisks={logic.scanRisks}
              groupedRisks={logic.groupedRisks}
            />
          </div>
        ) : (
          <div className="h-full overflow-y-auto">
            <CasesTab
              cases={logic.cases}
              onEdit={logic.openEdit}
              onDelete={logic.confirmDelete}
              onNavigate={(caseItem) => setActiveCaseId(caseItem.id)}
              searchQuery={logic.searchQuery}
              setSearchQuery={logic.setSearchQuery}
              filterType={logic.filterType}
              setFilterType={logic.setFilterType}
              filterStatus={logic.filterStatus}
              setFilterStatus={logic.setFilterStatus}
              sortType={logic.sortType}
              setSortType={logic.setSortType}
            />
          </div>
        )}
      </Window>

      <CaseEditModal
        open={logic.isNew || !!logic.editingCase}
        isNew={logic.isNew}
        isSaving={logic.isSaving}
        formData={logic.formData}
        setFormData={logic.setFormData}
        onClose={logic.closeModal}
        onSave={logic.handleSaveCase}
      />

      <DeleteConfirmModal
        target={logic.deleteTarget}
        entityLabel={t('clients.delete.entity')}
        onCancel={() => logic.setDeleteTarget(null)}
        onConfirm={logic.handleDeleteCase}
      />
    </div>
  );
}
