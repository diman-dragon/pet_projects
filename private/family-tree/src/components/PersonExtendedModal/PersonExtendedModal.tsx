import { useEffect, useMemo, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { Modal } from "@/components/Modal/Modal";
import { PersonList } from "@/components/PersonList/PersonList";
import { LockToggle } from "@/components/LockToggle/LockToggle";
import type { CanvasCardData } from "@/types/canvasCard";
import type { PersonData } from "@/types/person";
import type { RelationData } from "@/types/marriage";
import "@/components/PersonExtendedModal/PersonExtendedModal.css";

type MediaRecord = { media_type: string; file_path: string };
type Props = { card: CanvasCardData; cards: CanvasCardData[]; relations: RelationData[]; onSave: (person: PersonData) => Promise<void>; onDelete: () => void; onToggleLock: () => void; onClose: () => void };

export function PersonExtendedModal({ card, cards, relations, onSave, onDelete, onToggleLock, onClose }: Props) {
  const [draft, setDraft] = useState<PersonData>(card.person);
  const [media, setMedia] = useState<MediaRecord[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => { setDraft(card.person); void invoke<MediaRecord[]>("load_person_media", { personId: card.id }).then(setMedia).catch(() => setMedia([])); }, [card]);
  const setField = <K extends keyof PersonData>(field: K, value: PersonData[K]) => setDraft((current) => ({ ...current, [field]: value }));
  const parents = useMemo(() => cards.filter((candidate) => relations.some((relation) => relation.type === "parenthood" && relation.personBId === card.id && relation.personAId === candidate.id)), [cards, relations, card.id]);
  const children = useMemo(() => cards.filter((candidate) => relations.some((relation) => relation.type === "parenthood" && relation.personAId === card.id && relation.personBId === candidate.id)), [cards, relations, card.id]);
  const spouses = useMemo(() => cards.filter((candidate) => relations.some((relation) => relation.type === "marriage" && ((relation.personAId === card.id && relation.personBId === candidate.id) || (relation.personBId === card.id && relation.personAId === candidate.id)))), [cards, relations, card.id]);
  const documentGroups = useMemo(() => {
    const groups = new Map<string, string[]>();
    for (const item of media) groups.set(item.media_type, [...(groups.get(item.media_type) ?? []), item.file_path]);
    for (const relation of relations.filter((item) => item.personAId === card.id || item.personBId === card.id)) {
      const documents = Array.isArray(relation.details?.documents) ? relation.details.documents.filter((value): value is string => typeof value === "string") : [];
      if (documents.length) groups.set(`Связь · ${relation.type === "marriage" ? "брак" : "родительство"}`, [...(groups.get(`Связь · ${relation.type === "marriage" ? "брак" : "родительство"}`) ?? []), ...documents]);
    }
    return [...groups.entries()];
  }, [media, relations, card.id]);
  async function save() { if (card.locked) return; setSaving(true); setError(""); try { await onSave(draft); } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось сохранить данные персоны."); } finally { setSaving(false); } }
  async function addDocument() {
    const selected = await open({ multiple: false, directory: false });
    if (typeof selected !== "string") return;
    try { await invoke("import_person_media", { personId: card.id, sourcePath: selected, mediaType: "other" }); setMedia(await invoke<MediaRecord[]>("load_person_media", { personId: card.id })); } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось добавить документ."); }
  }
  const name = [draft.firstName, draft.patronymic, draft.lastName].filter(Boolean).join(" ") || "Персона";
  const labels: Array<[keyof PersonData, string, "input" | "textarea"]> = [["previousLastName", "Прежняя/девичья фамилия", "input"], ["genderIdentity", "Гендерная идентичность", "input"], ["nationality", "Национальность", "input"], ["ethnicity", "Этническая принадлежность", "input"], ["religion", "Религия", "input"], ["occupation", "Занятие", "input"], ["education", "Образование", "input"], ["description", "Краткие сведения", "textarea"], ["biography", "Биография", "textarea"], ["researchStatus", "Статус исследования", "input"]];
  return <Modal onClose={onClose} onBack={onClose} className="person-extended-modal" labelledBy="person-extended-title">
    <div className="person-extended-modal__header"><div><span>Персона · расширенные сведения</span><h2 id="person-extended-title">{name}</h2></div><button type="button" onClick={onClose} aria-label="Закрыть">×</button></div>
    {error && <div className="person-extended-modal__error" role="alert">{error}</div>}
    <section><h3>Основные данные</h3><div className="person-extended-modal__grid"><label>Имя<input disabled={card.locked} value={draft.firstName ?? ""} onChange={(event) => setField("firstName", event.target.value)} /></label><label>Отчество<input disabled={card.locked} value={draft.patronymic ?? ""} onChange={(event) => setField("patronymic", event.target.value)} /></label><label>Фамилия<input disabled={card.locked} value={draft.lastName ?? ""} onChange={(event) => setField("lastName", event.target.value)} /></label><label>Пол<select disabled={card.locked} value={draft.sex ?? ""} onChange={(event) => setField("sex", event.target.value ? event.target.value as PersonData["sex"] : undefined)}><option value="">Не указан</option><option value="male">Мужчина</option><option value="female">Женщина</option></select></label><label>Рождение<input disabled={card.locked} value={draft.birthDate ?? ""} onChange={(event) => setField("birthDate", event.target.value)} /></label><label>Смерть<input disabled={card.locked} value={draft.deathDate ?? ""} onChange={(event) => setField("deathDate", event.target.value)} /></label></div></section>
    <section><h3>Сведения и биография</h3><div className="person-extended-modal__grid">{labels.map(([field, label, kind]) => <label key={String(field)}>{label}{kind === "textarea" ? <textarea disabled={card.locked} value={String(draft[field] ?? "")} onChange={(event) => setField(field, event.target.value)} /> : <input disabled={card.locked} value={String(draft[field] ?? "")} onChange={(event) => setField(field, event.target.value)} />}</label>)}</div></section>
    <section><h3>Браки</h3>{spouses.length ? <PersonList cards={spouses} showGeneration={false} /> : <p>Нет зарегистрированных браков.</p>}</section>
    <section><h3>Родители</h3>{parents.length ? <PersonList cards={parents} showGeneration={false} /> : <p>Родители не указаны.</p>}</section>
    <section><h3>Дети</h3>{children.length ? <PersonList cards={children} showGeneration={false} /> : <p>Дети не указаны.</p>}</section>
    <section><div className="person-extended-modal__section-header"><h3>Документы и медиа</h3><button type="button" disabled={card.locked} onClick={() => void addDocument()}>Добавить документ</button></div>{documentGroups.length ? <div className="person-extended-modal__document-groups">{documentGroups.map(([type, paths]) => <div key={type}><h4>{type === "profile" ? "Фото профиля" : type === "birth" ? "Документы о рождении" : type === "death" ? "Документы о смерти" : type === "marriage" ? "Документы о браках" : type === "residence" ? "Документы о месте жительства" : type === "military" ? "Военные документы" : type}</h4><ul className="person-extended-modal__documents">{paths.map((path, index) => <li key={`${path}-${index}`}><small>{path}</small></li>)}</ul></div>)}</div> : <p>Документы не добавлены.</p>}</section>
    <div className="person-extended-modal__footer"><button type="button" className="person-extended-modal__delete" disabled={card.locked} onClick={onDelete}>Удалить</button><LockToggle locked={Boolean(card.locked)} onToggle={onToggleLock} /><div><button type="button" onClick={onClose}>Отмена</button><button type="button" className="person-extended-modal__primary" disabled={saving || card.locked} onClick={() => void save()}>{saving ? "Сохранение…" : "Сохранить"}</button></div></div>
  </Modal>;
}
