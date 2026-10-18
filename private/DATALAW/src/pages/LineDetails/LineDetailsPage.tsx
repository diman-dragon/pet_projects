// src/pages/LineDetails/LineDetailsPage.tsx
import { useLineDetailsLogic } from './useLineDetailsLogic';

import AppealDetailsPage from '../AppealDetails/AppealDetailsPage';

// Импортируем сценарии
import { LitigationLineScenario } from './scenarios/LitigationLineScenario';
import { ContractSectionScenario } from './scenarios/ContractSectionScenario';
import { ConsultationQuestionScenario } from './scenarios/ConsultationQuestionScenario';
import { OtherLineScenario } from './scenarios/OtherLineScenario';

// ИМПОРТИРУЕМ ID отсюда, чтобы TS видел его использование в интерфейсе Props
import { ID, Appeal } from './types';

interface LineDetailsPageProps {
  lineId: ID; // Теперь ID "прочитан" как тип для этого свойства
  onBack: () => void;
  activeAppealId: ID | null;
  setActiveAppealId: (id: ID | null) => void;
}

export default function LineDetailsPage({
  lineId,
  onBack,
  activeAppealId,
  setActiveAppealId,
}: LineDetailsPageProps) {
  // Передаем в логику. Если lineId это строка "123", Number() превратит её в число
  const logic = useLineDetailsLogic(Number(lineId));

  /**
   * ГЛУБОКАЯ НАВИГАЦИЯ (Уровень 3)
   * Если выбрано обращение — рендерим его страницу
   */
  if (activeAppealId) {
    return (
      <AppealDetailsPage appealId={Number(activeAppealId)} onBack={() => setActiveAppealId(null)} />
    );
  }

  const scenarioProps = {
    line: logic.line!,
    appeals: logic.appeals,
    loading: logic.loading,
    view: logic.view,
    setView: logic.setView,
    onBack,
    onNavigateAppeal: (appeal: Appeal) => setActiveAppealId(appeal.id),
    isNew: logic.isNew,
    editingAppeal: logic.editingAppeal,
    isSaving: logic.isSaving,
    formData: logic.formData,
    setFormData: logic.setFormData,
    openCreate: logic.openCreate,
    openEdit: logic.openEdit,
    closeModal: logic.closeModal,
    handleSaveAppeal: logic.handleSaveAppeal,
    deleteTarget: logic.deleteTarget,
    setDeleteTarget: logic.setDeleteTarget,
    handleDeleteAppeal: logic.handleDeleteAppeal,
    handleUpdateLine: logic.handleUpdateLine,
  };

  if (!logic.loading && logic.line && logic.case_) {
    const type = logic.case_.case_type;

    if (type === 'contract') {
      return <ContractSectionScenario {...scenarioProps} />;
    }

    if (type === 'consulting') {
      return <ConsultationQuestionScenario {...scenarioProps} />;
    }

    if (type === 'other') {
      return <OtherLineScenario {...scenarioProps} />;
    }

    return <LitigationLineScenario {...scenarioProps} />;
  }

  return <LitigationLineScenario {...scenarioProps} />;
}
