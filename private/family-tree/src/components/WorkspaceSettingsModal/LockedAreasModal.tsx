import { useState } from "react";
import { Modal } from "@/components/Modal/Modal";
import { ConfirmActionModal } from "@/components/ConfirmActionModal/ConfirmActionModal";
import { useCanvasCardsStore } from "@/store/useCanvasCardsStore";
import { GENERATION_HEIGHT } from "@/lib/canvas/generationConstants";
import { getLockedAreas, type LockedArea } from "@/lib/canvas/getLockedAreas";
import { removeLockedArea } from "@/lib/canvas/lockedAreaStorage";
import "@/components/WorkspaceSettingsModal/WorkspaceSettingsModal.css";

type LockedAreasModalProps = { onClose: () => void; onCompleted: () => void };
export function LockedAreasModal({ onClose, onCompleted }: LockedAreasModalProps) {
  const cards = useCanvasCardsStore((state) => state.cards);
  const setCardsLocked = useCanvasCardsStore((state) => state.setCardsLocked);
  const areas = getLockedAreas(cards);
  const [pendingArea, setPendingArea] = useState<LockedArea | null>(null);
  const [error, setError] = useState("");
  const [selectedAreaId, setSelectedAreaId] = useState<string | null>(null);
  const previewing = selectedAreaId !== null;
  function clearPreview() {
    window.dispatchEvent(new Event("family-tree:clear-locked-area-preview"));
    setSelectedAreaId(null);
  }
  function closeModal() {
    clearPreview();
    onClose();
  }

  function previewArea(area: LockedArea) {
    setSelectedAreaId(area.id);
    window.dispatchEvent(new CustomEvent("family-tree:focus-locked-area", { detail: { cardIds: area.cards.map((card) => card.id) } }));
  }

  async function unlockArea() {
    if (!pendingArea) return;
    try {
      await setCardsLocked(pendingArea.cards.map((card) => card.id), false);
      removeLockedArea(pendingArea.id);
      setPendingArea(null);
      clearPreview();
      onCompleted();
    } catch (unlockError) {
      setError(`Не удалось разблокировать область: ${String(unlockError)}`);
      setPendingArea(null);
    }
  }

  return <Modal onClose={previewing ? () => undefined : closeModal} onBack={previewing ? clearPreview : closeModal} backdropClassName={`workspace-settings-backdrop workspace-settings-backdrop--submodal${previewing ? " workspace-settings-backdrop--area-preview" : ""}`} className="workspace-settings-modal workspace-settings-modal--locks" labelledBy="locked-areas-title">
      <div className="workspace-settings-modal__header"><div><span className="workspace-settings-modal__eyebrow">Настройки · фиксация</span><h2 id="locked-areas-title">Заблокированные области</h2></div><button type="button" className="workspace-settings-modal__close" onClick={closeModal} aria-label="Закрыть">×</button></div>
      <p className="workspace-settings-modal__hint">Выберите область для просмотра. Канвас подберёт масштаб по всей области целиком, а не по отдельным карточкам.</p>
      {areas.length ? <div className="workspace-settings-modal__locked-areas">{areas.map((area, index) => { const firstCard = area.cards[0]; if (!firstCard) return null; return <button key={area.id} type="button" className={`workspace-settings-modal__locked-area${selectedAreaId === area.id ? " workspace-settings-modal__locked-area--active" : ""}`} onClick={() => previewArea(area)}><span className="workspace-settings-modal__locked-area-icon" aria-hidden="true">⌑</span><span><strong>Область {index + 1}</strong><small>{area.cards.length} {area.cards.length === 1 ? "карточка" : "карточек"} · поколение {Math.round(firstCard.y / GENERATION_HEIGHT)}</small></span><span aria-hidden="true">›</span></button>; })}</div> : <div className="workspace-settings-modal__empty">Заблокированных областей нет.</div>}
      {error && <div className="workspace-settings-modal__error" role="alert">{error}</div>}
      <div className="workspace-settings-modal__actions"><button type="button" className="workspace-settings-modal__secondary" onClick={closeModal}>Назад</button>{previewing && <button type="button" className="workspace-settings-modal__primary" onClick={() => { const area = areas.find((candidate) => candidate.id === selectedAreaId); if (area) setPendingArea(area); }}>Разблокировать область</button>}</div>
    {pendingArea && <ConfirmActionModal title="Разблокировать область?" message={`Фиксация будет снята с ${pendingArea.cards.length} карточек. После этого их можно будет снова перемещать.`} confirmLabel="Разблокировать" onClose={() => setPendingArea(null)} onConfirm={() => void unlockArea()} />}
  </Modal>;
}
