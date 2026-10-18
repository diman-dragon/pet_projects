import { Button } from "../../ui/Button";
import { Modal } from "../../ui/Modal";

interface ConfirmModalProps {
  open: boolean;
  title: string;
  message: string;
  confirmLabel: string;
  onConfirm: () => void;
  onClose: () => void;
}

export function ConfirmModal({
  open,
  title,
  message,
  confirmLabel,
  onConfirm,
  onClose
}: ConfirmModalProps) {
  return (
    <Modal
      isOpen={open}
      title={title}
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose} variant="ghost">Отмена</Button>
          <Button onClick={onConfirm} variant="danger">
            {confirmLabel}
          </Button>
        </>
      }
    >
      <p className="text-sm text-slate-600">{message}</p>
    </Modal>
  );
}
