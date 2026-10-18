import { Trash2 } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { Button } from '../form/Button';
import { AppModal } from './AppModal';
import { DeleteConfirmContent } from './DeleteConfirmContent';
import { ModalTitle } from './ModalTitle';

export function DeleteConfirmModal({
  target,
  entityLabel,
  entityName,
  note,
  onCancel,
  onConfirm,
}: {
  target: any | null;
  entityLabel?: string;
  entityName?: string;
  note?: string;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const { t } = useI18n();
  const resolvedLabel = entityLabel ?? t('delete.entityDefault');
  const name = entityName || target?.title || target?.name || t('delete.elementFallback');
  return (
    <AppModal open={!!target} onClose={onCancel} size="md" title={<ModalTitle icon={<Trash2 size={18} />} title={t('delete.modalTitle')} />} footer={<div className="flex justify-end gap-2"><Button variant="secondary" onClick={onCancel}>{t('common.cancel')}</Button><Button variant="danger" icon={<Trash2 size={16} />} onClick={onConfirm}>{t('delete.confirm')}</Button></div>}>
      <DeleteConfirmContent entityLabel={resolvedLabel} entityName={name} note={note} />
    </AppModal>
  );
}
