// src/pages/CaseDetails/CaseDetailsPage.tsx
import { useCaseDetailsLogic } from './useCaseDetailsLogic';

// ИМПОРТИРУЕМ страницу деталей линии и обращения
import LineDetailsPage from '../LineDetails/LineDetailsPage';
import AppealDetailsPage from '../AppealDetails/AppealDetailsPage';

// Импортируем сценарии
import { LitigationScenario } from './scenarios/LitigationScenario';
import { ContractScenario } from './scenarios/ContractScenario';
import { ConsultationScenario } from './scenarios/ConsultationScenario';
import { OtherScenario } from './scenarios/OtherScenario';

interface CaseDetailsPageProps {
  caseId: string | number;
  onBack: () => void;
  // Сделали пропсы обязательными для корректного проброса вглубь
  activeLineId: string | number | null;
  setActiveLineId: (id: string | number | null) => void;
  activeAppealId: string | number | null;
  setActiveAppealId: (id: string | number | null) => void;
}

export default function CaseDetailsPage({
  caseId,
  onBack,
  activeLineId,
  setActiveLineId,
  activeAppealId,
  setActiveAppealId,
}: CaseDetailsPageProps) {
  const logic = useCaseDetailsLogic(Number(caseId));

  // 0. ГЛУБОКАЯ НАВИГАЦИЯ (Уровень 4)
  // Если выбрано конкретное обращение/документ — рендерим его страницу
  if (activeAppealId) {
    return (
      <AppealDetailsPage
        appealId={Number(activeAppealId)}
        onBack={() => setActiveAppealId(null)}
      />
    );
  }

  // 1. ГЛУБОКАЯ НАВИГАЦИЯ (Уровень 3)
  // Если в стейте есть выбранная линия — проваливаемся в неё
  // Исключение: для "плоских" типов (консультация, договор, прочее, судебное) всегда остаемся на уровне дела
  const isFlatType = ['consulting', 'contract', 'other', 'litigation'].includes(logic.case_?.case_type || '');

  if (activeLineId && !isFlatType) {
    return (
      <LineDetailsPage
        lineId={activeLineId}
        onBack={() => {
          setActiveLineId(null);
          setActiveAppealId(null); // Важно очистить всё при возврате к делу
        }}
        activeAppealId={activeAppealId}
        setActiveAppealId={setActiveAppealId}
      />
    );
  }

  const scenarioProps = {
    case_: logic.case_,
    loading: logic.loading,
    view: logic.view,
    setView: logic.setView,
    lines: logic.lines,
    onBack,
    onNavigateLine: (line: any) => setActiveLineId(line.id),
    onNavigateAppeal: (appeal: any) => {
      if (logic.defaultLine) {
        setActiveLineId(logic.defaultLine.id);
        setActiveAppealId(appeal.id);
      }
    },
    isEditOpen: logic.isEditOpen,
    isSaving: logic.isSaving,
    formData: logic.formData,
    setFormData: logic.setFormData,
    openCreate: logic.openCreate,
    openEdit: logic.openEdit,
    closeEdit: logic.closeEdit,
    handleSave: logic.handleSave,
    isDeleteOpen: logic.isDeleteOpen,
    selectedItem: logic.selectedItem,
    openDelete: logic.openDelete,
    closeDelete: logic.closeDelete,
    confirmDelete: logic.confirmDelete,
    // New props for consulting/other
    defaultLine: logic.defaultLine,
    appeals: logic.appeals,
    handleUpdateCase: logic.handleUpdateCase,
    handleUpdateDefaultLine: logic.handleUpdateDefaultLine,
    handleSaveAppeal: logic.handleSaveAppeal,
    handleUploadDocument: logic.handleUploadDocument,
    handleDeleteAppeal: logic.handleDeleteAppeal,
  };

  if (!logic.loading && logic.case_) {
    const type = logic.case_.case_type;

    if (type === 'contract') {
      return <ContractScenario {...scenarioProps} />;
    }

    if (type === 'consulting') {
      return <ConsultationScenario {...scenarioProps} />;
    }

    if (type === 'other') {
      return <OtherScenario {...scenarioProps} />;
    }

    // Default to Litigation for litigation, administrative, criminal
    return <LitigationScenario {...scenarioProps} />;
  }

  // Пока грузится — показываем дефолтный пустой экран с лоадером (можно взять из любого сценария)
  return <LitigationScenario {...scenarioProps} />;
}
