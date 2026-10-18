import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Modal } from "@/components/Modal/Modal";
import { ShortModalActions } from "@/components/ShortModalActions/ShortModalActions";
import { LockToggle } from "@/components/LockToggle/LockToggle";
import { PhotoCropModal } from "@/components/PhotoCropModal/PhotoCropModal";
import { usePhotoSrc } from "@/lib/person/usePhotoSrc";
import type { PersonData } from "@/types/person";
import "@/components/PersonShortModal/PersonShortModal.css";

type Props = {
  personId: string;
  person: PersonData;
  sourceKey: string;
  isNew: boolean;
  locked?: boolean;
  onSave: (person: PersonData, photoBytes?: number[]) => Promise<void>;
  onDelete: () => void;
  onExtendedEdit?: () => void;
  onToggleLock?: () => void;
  onClose: () => void;
};

function arePersonsEqual(first: PersonData, second: PersonData) {
  return JSON.stringify(first) === JSON.stringify(second);
}

function withPhoto(person: PersonData, photoUrl?: string): PersonData {
  return {
    ...person,
    photoUrl,
    photoZoom: photoUrl ? 1 : undefined,
    photoOffsetX: photoUrl ? 0 : undefined,
    photoOffsetY: photoUrl ? 0 : undefined,
  };
}

export function PersonShortModal({ personId, person, sourceKey, isNew, locked = false, onSave, onDelete, onExtendedEdit, onToggleLock, onClose }: Props) {
  const [draft, setDraft] = useState<PersonData>(person);
  const [savedDraft, setSavedDraft] = useState<PersonData>(person);
  const [saving, setSaving] = useState(false);
  const [showPhotoModal, setShowPhotoModal] = useState(false);
  const [photoError, setPhotoError] = useState("");
  const pendingPhotoBytes = useRef<number[] | null>(null);
  const previewSrc = usePhotoSrc(draft.photoUrl);

  useEffect(() => {
    setDraft(person);
    setSavedDraft(person);
    pendingPhotoBytes.current = null;
    setPhotoError("");
  }, [sourceKey]);

  const fullName = [draft.firstName, draft.patronymic, draft.lastName].filter(Boolean).join(" ");
  const hasChanges = !arePersonsEqual(draft, savedDraft) || pendingPhotoBytes.current !== null;

  function setField<K extends keyof PersonData>(field: K, value: PersonData[K]) {
    setDraft((current) => ({ ...current, [field]: value }));
  }

  async function savePerson() {
    if (locked) return;
    setSaving(true);
    setPhotoError("");
    try {
      const bytes = pendingPhotoBytes.current ?? undefined;
      const payload = bytes ? withPhoto({ ...draft, photoUrl: undefined }) : draft;
      await onSave(payload, bytes);
      pendingPhotoBytes.current = null;
      setSavedDraft(draft);
    } catch (reason) {
      setPhotoError(reason instanceof Error ? reason.message : "Не удалось сохранить карточку.");
    } finally {
      setSaving(false);
    }
  }

  async function handleSavePhoto(imageData: number[]) {
    setPhotoError("");
    if (!personId) {
      pendingPhotoBytes.current = imageData;
      const preview = URL.createObjectURL(new Blob([new Uint8Array(imageData)], { type: "image/png" }));
      const next = withPhoto(draft, preview);
      setDraft(next);
      return;
    }
    const result = await invoke<string>("save_person_photo_crop", {
      personId,
      imageData,
      zoom: 1,
      offsetX: 0,
      offsetY: 0,
    });
    pendingPhotoBytes.current = null;
    const next = withPhoto(draft, result);
    setDraft(next);
    setSavedDraft(next);
    await onSave(next);
  }

  async function handleDeletePhoto() {
    setPhotoError("");
    pendingPhotoBytes.current = null;
    if (personId) {
      await invoke("delete_person_photo", { personId });
      const next = withPhoto(draft, undefined);
      setDraft(next);
      setSavedDraft(next);
      await onSave(next);
      return;
    }
    setDraft(withPhoto(draft, undefined));
  }

  return <Modal onClose={showPhotoModal ? () => setShowPhotoModal(false) : onClose} onBack={showPhotoModal ? () => setShowPhotoModal(false) : onClose} className="person-short-modal" labelledBy="person-short-title">
    <div className="person-short-modal__header"><div><span className="person-short-modal__eyebrow">Персона</span><h2 id="person-short-title">{fullName || "Персона"}</h2></div><button type="button" className="person-short-modal__close" onClick={onClose} aria-label="Закрыть">×</button></div>
    {photoError && <p className="person-short-modal__photo-error" role="alert">{photoError}</p>}
    <div className="person-short-modal__photo-section">
      {draft.photoUrl ? (
        <div className="person-short-modal__photo-preview">
          {previewSrc ? <img src={previewSrc} alt="Фото" /> : <div className="person-short-modal__photo-placeholder" />}
          <div className="person-short-modal__photo-actions">
            <button type="button" disabled={locked} onClick={() => setShowPhotoModal(true)}>Изменить</button>
            <button type="button" disabled={locked} onClick={() => void handleDeletePhoto().catch((reason) => setPhotoError(String(reason)))}>Удалить</button>
          </div>
        </div>
      ) : (
        <button type="button" className="person-short-modal__add-photo" disabled={locked} onClick={() => setShowPhotoModal(true)}>Добавить фото</button>
      )}
    </div>
    <div className="person-short-modal__grid">
      <label>Имя<input disabled={locked} autoFocus value={draft.firstName ?? ""} onChange={(event) => setField("firstName", event.target.value)} /></label>
      <label>Отчество<input disabled={locked} value={draft.patronymic ?? ""} onChange={(event) => setField("patronymic", event.target.value)} /></label>
      <label>Фамилия<input disabled={locked} value={draft.lastName ?? ""} onChange={(event) => setField("lastName", event.target.value)} /></label>
      <label>Пол<select disabled={locked} value={draft.sex ?? ""} onChange={(event) => setField("sex", event.target.value ? event.target.value as PersonData["sex"] : undefined)}><option value="">Не указан</option><option value="male">Мужчина</option><option value="female">Женщина</option></select></label>
      <label>Дата рождения<input disabled={locked} placeholder="01.01.2000" value={draft.birthDate ?? ""} onChange={(event) => setField("birthDate", event.target.value)} /></label>
      <label>Дата смерти<input disabled={locked} placeholder="01.01.2000" value={draft.deathDate ?? ""} onChange={(event) => setField("deathDate", event.target.value)} /></label>
    </div>
    <div className="person-short-modal__lock-row">{!isNew && onToggleLock && <LockToggle locked={locked} onToggle={onToggleLock} />}</div>
    <ShortModalActions
      isNew={isNew}
      saving={saving}
      submitLabel={isNew ? "Создать" : hasChanges ? "Сохранить изменения" : "Редактировать"}
      onCancel={onClose}
      onSubmit={() => void (isNew || hasChanges ? savePerson() : onExtendedEdit?.())}
      onDelete={locked ? undefined : onDelete}
    />
    {showPhotoModal && (
      <PhotoCropModal
        title="Фото человека"
        currentUrl={draft.photoUrl}
        initialZoom={1}
        initialOffsetX={0}
        initialOffsetY={0}
        onSave={async (imageData) => {
          await handleSavePhoto(imageData);
          setShowPhotoModal(false);
        }}
        onClose={() => setShowPhotoModal(false)}
      />
    )}
  </Modal>;
}
