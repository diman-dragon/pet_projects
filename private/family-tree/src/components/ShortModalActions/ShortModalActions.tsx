import "@/components/ShortModalActions/ShortModalActions.css";

type Props = {
  isNew: boolean;
  saving?: boolean;
  submitLabel?: string;
  onCancel: () => void;
  onSubmit: () => void;
  onDelete?: () => void;
};

export function ShortModalActions({ isNew, saving = false, submitLabel, onCancel, onSubmit, onDelete }: Props) {
  return <div className="short-modal-actions">
    {onDelete && !isNew && <button type="button" className="short-modal-actions__delete" onClick={onDelete}>Удалить</button>}
    <div className="short-modal-actions__buttons">
      <button type="button" className="short-modal-actions__secondary" onClick={onCancel}>Отмена</button>
      <button type="button" className="short-modal-actions__primary" disabled={saving} onClick={onSubmit}>{saving ? "Сохранение…" : submitLabel ?? (isNew ? "Создать" : "Редактировать")}</button>
    </div>
  </div>;
}
