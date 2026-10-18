import { Modal as BaseModal } from "../../ui/Modal";
import type { ReactNode } from "react";

interface ModalProps {
  open: boolean;
  title: string;
  description?: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
}

export function Modal({
  open,
  title,
  description,
  onClose,
  children,
  footer
}: ModalProps) {
  return (
    <BaseModal
      isOpen={open}
      title={title}
      onClose={onClose}
      footer={footer}
    >
      {description && <p className="mb-4 text-sm text-gray-500">{description}</p>}
      {children}
    </BaseModal>
  );
}
