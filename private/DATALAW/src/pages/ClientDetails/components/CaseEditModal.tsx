import { Briefcase, Plus, Save } from 'lucide-react';
import { Button } from '../../../components/ui/form/Button';
import { RequiredFieldsLegend } from '../../../components/ui/form/RequiredFieldsLegend';
import { ModalTitle } from '../../../components/ui/overlays/ModalTitle';
import { Modal } from '../../../components/ui/overlays/Modal';
import type { Case } from '../../../types/entities';
import { CaseEditForm } from './CaseEditForm';
import { useI18n } from '../../../i18n/AppLanguageProvider';

interface Props {
  open: boolean;
  isNew: boolean;
  isSaving: boolean;
  formData: Partial<Case>;
  setFormData: (data: Partial<Case>) => void;
  onClose: () => void;
  onSave: () => void;
}

export function CaseEditModal({
  open,
  isNew,
  isSaving,
  formData,
  setFormData,
  onClose,
  onSave,
}: Props) {
  const { t } = useI18n();
  const caseType = formData.case_type ?? '';
  const scenarioInfo = (() => {
    switch (caseType) {
      case 'contract':
        return {
          title: 'Договорная работа',
          description:
            'Структура дела: год → контрагент → Documents. Обязательно заполните контрагента для этого сценария.',
        };
      case 'litigation':
        return {
          title: 'Судебный спор',
          description:
            'Стандартный судебный сценарий создаёт корневую папку дела с разделом 00_BASE и 01_Proceedings для линий.',
        };
      case 'consulting':
        return {
          title: 'Консультирование / Заключение',
          description:
            'Консультационный сценарий хранит материалы дела в основном каталоге проекта без дополнительной контрагенсткой структуры.',
        };
      default:
        return {
          title: 'Выберите вид работы',
          description:
            'После выбора типа проекта модальное окно адаптируется под соответствующий сценарий и поля.',
        };
    }
  })();

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={
        <ModalTitle
          icon={<Briefcase size={18} />}
          title={isNew ? t('case.modal.createTitle') : t('case.modal.editTitle')}
          subtitle={t('case.modal.editTitle')}
        />
      }
      footer={
        <div className="flex w-full items-center justify-end gap-2">
          <Button variant="ghost" onClick={onClose} disabled={isSaving}>
            {t('common.cancel')}
          </Button>
          <Button
            variant="primary"
            loading={isSaving}
            disabled={
              !formData.title?.trim() ||
              (formData.case_type === 'contract' && !formData.counterparty?.trim())
            }
            onClick={onSave}
            icon={isNew ? <Plus size={16} /> : <Save size={16} />}
            className="min-w-[140px]"
          >
            {isNew ? t('case.modal.create') : t('case.modal.save')}
          </Button>
        </div>
      }
    >
      <div className="space-y-4">
        <RequiredFieldsLegend />
        <div className="rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface-soft)] p-4">
          <div className="text-[length:var(--text-sm)] font-semibold text-[var(--app-text)]">{scenarioInfo.title}</div>
          <div className="mt-1 text-[13px] text-[var(--app-text-secondary)]">
            {scenarioInfo.description}
          </div>
        </div>
        <CaseEditForm formData={formData} setFormData={setFormData} isNew={isNew} />
      </div>
    </Modal>
  );
}
