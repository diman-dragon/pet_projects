import { useEffect, useRef, type ReactNode } from "react";
import "@/components/Modal/Modal.css";

type ModalProps = {
  children: ReactNode;
  onClose: () => void;
  onBack: () => void;
  backdropClassName?: string;
  className?: string;
  role?: "dialog" | "alertdialog";
  labelledBy?: string;
};

export function Modal({ children, onClose, onBack, backdropClassName = "", className = "", role = "dialog", labelledBy }: ModalProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onCloseRef.current();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => { document.removeEventListener("keydown", handleKeyDown); document.body.style.overflow = previousOverflow; };
  }, []);

  return (
    <div className={`app-modal-backdrop ${backdropClassName}`} onMouseDown={(event) => { event.stopPropagation(); if (event.target === event.currentTarget) onClose(); }} onPointerDown={(event) => { event.stopPropagation(); if (event.target === event.currentTarget) onClose(); }} onWheel={(event) => event.stopPropagation()} onPointerMove={(event) => event.stopPropagation()}>
      <div ref={contentRef} tabIndex={-1} className={`app-modal ${className}`} role={role} aria-modal="true" aria-labelledby={labelledBy} onMouseDown={(event) => event.stopPropagation()}>
        <button type="button" className="app-modal__back" onClick={onBack} aria-label="Назад">← Назад</button>
        {children}
      </div>
    </div>
  );
}
