import { useEffect, useState } from "react";
import { Modal } from "@/components/Modal/Modal";
import { PersonList } from "@/components/PersonList/PersonList";
import { ShortModalActions } from "@/components/ShortModalActions/ShortModalActions";
import type { CanvasCardData } from "@/types/canvasCard";
import type { RelationType } from "@/types/marriage";
import "@/components/RelationShortModal/RelationShortModal.css";

type Props = { type: RelationType; cards: CanvasCardData[]; parentIds?: string[]; isNew: boolean; status?: "active" | "dissolved"; autoAlign?: boolean; onAutoAlignChange?: (value: boolean) => Promise<void>; onSave: (status: "active" | "dissolved") => Promise<void>; onDelete: () => void; onExtendedEdit?: () => void; onClose: () => void };
export function RelationShortModal({ type, cards, parentIds = [], isNew, status = "active", autoAlign = true, onAutoAlignChange, onSave, onDelete, onExtendedEdit, onClose }: Props) {
  const [saving, setSaving] = useState(false);
  const [selectedStatus, setSelectedStatus] = useState(status);
  useEffect(() => setSelectedStatus(status), [isNew, status]);
  const title = type === "marriage" ? "Брак" : "Родительская связь";
  const roles = Object.fromEntries(cards.map((card) => {
    if (type === "marriage") return [card.id, undefined];
    if (!parentIds.includes(card.id)) return [card.id, "Ребенок"];
    return [card.id, card.person.sex === "female" ? "Мама" : card.person.sex === "male" ? "Папа" : "Родитель"];
  }));
  async function save() { setSaving(true); try { await onSave(selectedStatus); } finally { setSaving(false); } }
  return <Modal onClose={onClose} onBack={onClose} className="relation-short-modal" labelledBy="relation-short-title">
    <div className="relation-short-modal__header"><div><span className="relation-short-modal__eyebrow">Связь</span><h2 id="relation-short-title">{title}</h2></div><button type="button" className="relation-short-modal__close" onClick={onClose} aria-label="Закрыть">×</button></div>
    {type === "marriage" && <div className="relation-short-modal__status"><span>Статус</span>{isNew ? <div className="relation-short-modal__status-buttons" role="group" aria-label="Статус брака"><button type="button" className={selectedStatus === "active" ? "is-active" : ""} aria-pressed={selectedStatus === "active"} onClick={() => setSelectedStatus("active")}>Действует</button><button type="button" className={selectedStatus === "dissolved" ? "is-active" : ""} aria-pressed={selectedStatus === "dissolved"} onClick={() => setSelectedStatus("dissolved")}>Расторгнут</button></div> : <strong>{status === "dissolved" ? "Расторгнут" : "Действует"}</strong>}</div>}
    {type === "parenthood" && <div className="relation-short-modal__alignment"><span>Центрирование детей</span><div className="relation-short-modal__alignment-buttons"><button type="button" className={autoAlign ? "is-active" : ""} aria-pressed={autoAlign} onClick={() => void onAutoAlignChange?.(true)}>Авто</button><button type="button" className={!autoAlign ? "is-active" : ""} aria-pressed={!autoAlign} onClick={() => void onAutoAlignChange?.(false)}>Свободно</button></div></div>}
    <div className="relation-short-modal__people"><span>Участники</span><PersonList cards={cards} roles={roles} /></div>
    <ShortModalActions isNew={isNew} saving={saving} onCancel={onClose} onSubmit={() => void (isNew ? save() : onExtendedEdit?.())} onDelete={onDelete} />
  </Modal>;
}
