import { Modal } from "@/components/Modal/Modal";
import "@/components/ConfirmActionModal/ConfirmActionModal.css";

type ConfirmActionModalProps = { title: string; message: string; confirmLabel?: string; onConfirm: () => void; onClose: () => void };

export function ConfirmActionModal({ title, message, confirmLabel = "Подтвердить", onConfirm, onClose }: ConfirmActionModalProps) {
  return (
    <Modal onClose={onClose} onBack={onClose} backdropClassName="confirm-action-backdrop" className="confirm-action-modal" role="alertdialog" labelledBy="confirm-action-title">
        <h2 id="confirm-action-title">{title}</h2>
        <p>{message}</p>
        <div className="confirm-action-modal__actions"><button type="button" className="confirm-action-modal__secondary" onClick={onClose}>Отмена</button><button type="button" className="confirm-action-modal__danger" onClick={onConfirm}>{confirmLabel}</button></div>
    </Modal>
  );
}
