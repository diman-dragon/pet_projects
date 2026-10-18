// src/pages/users/components/UserEditModal.tsx

import { UserPlus, Settings2 } from 'lucide-react';
import { AppModal } from '../../../components/ui/overlays/AppModal';
import { AppButton } from '../../../components/ui/form/AppButton';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import UserEditForm from './UserEditForm';

interface Props {
  open: boolean;
  onClose: () => void;
  logic: {
    selectedItem: any;
    formData: any;
    setFormData: (d: any) => void;
    handleSave: () => void;
    canAssignRole?: (role: string) => boolean;
  };
}

export function UserEditModal({ open, onClose, logic }: Props) {
  const { t } = useI18n();
  const isEdit = !!logic.selectedItem;

  return (
    <AppModal
      open={open}
      size="md"
      onClose={onClose}
      title={
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-[10px] bg-[var(--app-accent)]/[0.08] flex items-center justify-center text-[var(--app-accent)]">
            {isEdit ? <Settings2 size={18} /> : <UserPlus size={18} />}
          </div>
          <span className="text-[15px] font-bold text-[var(--app-text)]">
            {isEdit ? t('users.modal.editTitle') : t('users.modal.createTitle')}
          </span>
        </div>
      }
      footer={
        <>
          <AppButton variant="ghost" onClick={onClose}>
            {t('common.cancel')}
          </AppButton>
          <AppButton variant="primary" onClick={logic.handleSave}>
            {isEdit ? t('users.modal.save') : t('users.modal.create')}
          </AppButton>
        </>
      }
    >
      <div className="py-2">
        <UserEditForm
          formData={logic.formData}
          setFormData={logic.setFormData}
          canAssignRole={logic.canAssignRole}
        />
      </div>
    </AppModal>
  );
}
