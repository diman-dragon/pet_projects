import { AlertCircle } from 'lucide-react';
import { Button } from '../form/Button';
import { AppModal } from './AppModal';
import { ModalTitle } from './ModalTitle';

export function ConfirmModal({
  open,
  title,
  message,
  confirmText = 'Подтвердить',
  cancelText = 'Отмена',
  isDangerous = false,
  onCancel,
  onConfirm,
  isLoading = false,
}: {
  open: boolean;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  isDangerous?: boolean;
  onCancel: () => void;
  onConfirm: () => void | Promise<void>;
  isLoading?: boolean;
}) {
  return (
    <AppModal open={open} onClose={onCancel} size="md" title={<ModalTitle icon={<AlertCircle size={18} />} title={title} />} footer={<div className="flex justify-end gap-2"><Button variant="secondary" onClick={onCancel} disabled={isLoading}>{cancelText}</Button><Button variant={isDangerous ? 'danger' : 'primary'} onClick={onConfirm} loading={isLoading}>{confirmText}</Button></div>}>
      <p className="text-[length:var(--text-sm)] leading-[var(--leading-body)] text-[var(--app-text)]">{message}</p>
    </AppModal>
  );
}
