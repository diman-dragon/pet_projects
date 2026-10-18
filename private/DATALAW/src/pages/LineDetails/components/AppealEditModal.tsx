import { useI18n } from '../../../i18n/AppLanguageProvider';
import { FileText, Plus, Save } from 'lucide-react';
import { Button } from '../../../components/ui/form/Button';
import { RequiredFieldsLegend } from '../../../components/ui/form/RequiredFieldsLegend';
import { ModalTitle } from '../../../components/ui/overlays/ModalTitle';
import { Modal } from '../../../components/ui/overlays/Modal';
import { AppealEditForm } from './AppealEditForm';
import type { Appeal } from '../../../types/entities';

interface Props {
  open: boolean;
  isNew: boolean;
  isSaving: boolean;
  formData: Partial<Appeal>;
  setFormData: (data: Partial<Appeal>) => void;
  onClose: () => void;
  onSave: () => void;
  titleOverride?: string;
  subtitleOverride?: string;
  isLitigation?: boolean; // For litigation, don't show file upload
}

export function AppealEditModal({
  open,
  isNew,
  isSaving,
  formData,
  setFormData,
  onClose,
  onSave,
  titleOverride,
  subtitleOverride,
  isLitigation = false,
}: Props) {
  const { t } = useI18n();
  const title = titleOverride || (isNew ? t('appeal.modal.createTitle') : t('appeal.modal.editTitle'));
  const subtitle = subtitleOverride || t('appeal.modal.subtitle');

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={
        <ModalTitle
          icon={<FileText size={18} />}
          title={title}
          subtitle={subtitle}
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
            disabled={!formData.title?.trim()}
            onClick={onSave}
            icon={isNew ? <Plus size={16} /> : <Save size={16} />}
            className="min-w-[140px]"
          >
            {isNew ? t('appeal.modal.create') : t('appeal.modal.save')}
          </Button>
        </div>
      }
    >
      <div className="space-y-4">
        <RequiredFieldsLegend />
        <AppealEditForm formData={formData} setFormData={setFormData} isNew={isNew} isLitigation={isLitigation} />
      </div>
    </Modal>
  );
}
