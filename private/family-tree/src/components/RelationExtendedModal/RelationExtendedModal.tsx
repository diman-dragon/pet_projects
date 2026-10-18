import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { Modal } from "@/components/Modal/Modal";
import { PersonList } from "@/components/PersonList/PersonList";
import type { CanvasCardData } from "@/types/canvasCard";
import type { RelationData } from "@/types/marriage";
import { invoke } from "@tauri-apps/api/core";
import "@/components/RelationExtendedModal/RelationExtendedModal.css";

type Props = { relation: RelationData; cards: CanvasCardData[]; parentIds?: string[]; onSave: (details: Record<string, unknown>) => Promise<void>; onDelete: () => void; onClose: () => void };
const fields: Array<[string, string]> = [["registrationDate", "Дата регистрации"], ["registrationPlace", "Место регистрации"], ["registrationAuthority", "Орган регистрации"], ["recordNumber", "Номер записи"], ["dissolutionDate", "Дата расторжения"], ["dissolutionPlace", "Место расторжения"], ["dissolutionAuthority", "Орган расторжения"], ["notes", "Заметки"]];
export function RelationExtendedModal({ relation, cards, parentIds = [], onSave, onDelete, onClose }: Props) {
  const [draft, setDraft] = useState<Record<string, unknown>>(relation.details ?? {}); const [saving, setSaving] = useState(false); const [error, setError] = useState("");
  useEffect(() => setDraft(relation.details ?? {}), [relation]);
  const setField = (field: string, value: string) => setDraft((current) => ({ ...current, [field]: value || undefined }));
  const roles = Object.fromEntries(cards.map((card) => [card.id, relation.type === "parenthood" ? (parentIds.includes(card.id) ? (card.person.sex === "female" ? "Мама" : card.person.sex === "male" ? "Папа" : "Родитель") : "Ребенок") : undefined]));
  async function save() { setSaving(true); setError(""); try { await onSave(draft); } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось сохранить связь."); } finally { setSaving(false); } }
  async function addDocument() {
    const selected = await open({ multiple: false, directory: false }); if (typeof selected !== "string") return;
    try { const copiedPath = await invoke<string>("import_person_media", { personId: relation.personAId, sourcePath: selected, mediaType: relation.type === "marriage" ? "marriage" : "other" }); const documents = Array.isArray(draft.documents) ? draft.documents.filter((value): value is string => typeof value === "string") : []; setDraft((current) => ({ ...current, documents: [...documents, copiedPath] })); } catch (reason) { setError(String(reason)); }
  }
  async function swapMembers() {
    if (relation.type !== "marriage") return;
    try { await invoke("swap_relation_members", { relationId: relation.id }); onClose(); } catch (reason) { setError(String(reason)); }
  }
  async function deletePhoto() {
    if (relation.type !== "marriage") return;
    try { await invoke("delete_relation_photo", { relationId: relation.id }); const newDetails = { ...draft }; delete newDetails.couplePhotoUrl; setDraft(newDetails); await onSave(newDetails); } catch (reason) { setError(String(reason)); }
  }
  const name = relation.type === "marriage" ? "Брак" : "Родительская связь";
  return <Modal onClose={onClose} onBack={onClose} className="relation-extended-modal" labelledBy="relation-extended-title">
    <div className="relation-extended-modal__header"><div><span>Связь · расширенные сведения</span><h2 id="relation-extended-title">{name}</h2></div><button type="button" onClick={onClose} aria-label="Закрыть">×</button></div>
    {error && <div className="relation-extended-modal__error" role="alert">{error}</div>}
    <section><h3>Участники</h3><PersonList cards={cards} roles={roles} />{relation.type === "marriage" && <><button type="button" className="relation-extended-modal__swap" onClick={() => void swapMembers()}>Поменять супругов местами</button>{typeof draft.couplePhotoUrl === "string" && <button type="button" className="relation-extended-modal__swap" onClick={() => void deletePhoto()}>Удалить фото пары</button>}</>}</section>
    {relation.type === "marriage" && <label className="relation-extended-modal__status">Статус<select value={String(draft.status ?? "active")} onChange={(event) => setField("status", event.target.value)}><option value="active">Действует</option><option value="dissolved">Расторгнут</option></select></label>}
    <section><h3>Сведения о связи</h3><div className="relation-extended-modal__grid">{fields.map(([field, label]) => <label key={field}>{label}{field === "notes" ? <textarea value={String(draft[field] ?? "")} onChange={(event) => setField(field, event.target.value)} /> : <input value={String(draft[field] ?? "")} onChange={(event) => setField(field, event.target.value)} />}</label>)}</div></section>
    <section><div className="relation-extended-modal__section-header"><h3>Документы связи</h3><button type="button" onClick={() => void addDocument()}>Добавить документ</button></div>{Array.isArray(draft.documents) && draft.documents.length ? <ul>{draft.documents.map((document, index) => <li key={`${String(document)}-${index}`}>{String(document)}</li>)}</ul> : <p>Документы связи не добавлены.</p>}</section>
    <div className="relation-extended-modal__footer"><button type="button" className="relation-extended-modal__delete" onClick={onDelete}>Удалить</button><div><button type="button" onClick={onClose}>Отмена</button><button type="button" className="relation-extended-modal__primary" disabled={saving} onClick={() => void save()}>{saving ? "Сохранение…" : "Сохранить"}</button></div></div>
  </Modal>;
}
