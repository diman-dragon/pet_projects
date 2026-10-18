import type { ReactElement } from 'react';
import { AlertCircle, CheckCircle2, Info } from 'lucide-react';
import { Button } from '../form/Button';
import { AppModal } from './AppModal';
import { ModalTitle } from './ModalTitle';

type AlertType = 'error' | 'success' | 'info';
const icons = {
  error: <AlertCircle size={18} />,
  success: <CheckCircle2 size={18} />,
  info: <Info size={18} />,
} satisfies Record<AlertType, ReactElement>;

export function AlertModal({
  open,
  type = 'error',
  title,
  message,
  buttonText = 'OK',
  onClose,
}: {
  open: boolean;
  type?: AlertType;
  title: string;
  message: string;
  buttonText?: string;
  onClose: () => void;
}) {
  return (
    <AppModal open={open} onClose={onClose} size="md" title={<ModalTitle icon={icons[type]} title={title} />} footer={<div className="flex justify-end"><Button variant={type === 'error' ? 'danger' : 'primary'} onClick={onClose}>{buttonText}</Button></div>}>
      <p className="text-[length:var(--text-sm)] leading-[var(--leading-body)] text-[var(--app-text)]">{message}</p>
    </AppModal>
  );
}
