import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Modal } from "@/components/Modal/Modal";
import { ConfirmActionModal } from "@/components/ConfirmActionModal/ConfirmActionModal";
import { useCanvasCardsStore } from "@/store/useCanvasCardsStore";

type DangerousActionsModalProps = { onClose: () => void; onCompleted: () => void };
type Action = "clear_tree" | "clear_relations" | null;
export function DangerousActionsModal({ onClose, onCompleted }: DangerousActionsModalProps) {
  const [confirmation, setConfirmation] = useState<Action>(null); const [error, setError] = useState("");
  async function runAction(action: Action) {
    if (!action) return;
    try {
      if (action === "clear_tree") {
        await invoke("clear_tree");
      } else if (action === "clear_relations") {
        await invoke("clear_relations");
      }
      setConfirmation(null);
      await useCanvasCardsStore.getState().loadGraph();
      onCompleted();
    } catch (actionError) { setError(`Не удалось выполнить действие: ${String(actionError)}`); setConfirmation(null); }
  }
  return <Modal onClose={onClose} onBack={onClose} backdropClassName="workspace-settings-backdrop workspace-settings-backdrop--submodal" className="workspace-settings-modal workspace-settings-modal--danger" labelledBy="danger-actions-title">
    <div className="workspace-settings-modal__header"><div><span className="workspace-settings-modal__eyebrow">Настройки · осторожно</span><h2 id="danger-actions-title">Опасные действия</h2></div><button type="button" className="workspace-settings-modal__close" onClick={onClose} aria-label="Закрыть">×</button></div><p className="workspace-settings-modal__hint">Эти операции изменяют данные дерева и требуют отдельного подтверждения.</p>
    <div className="workspace-settings-modal__danger-zone">
      <button type="button" className="workspace-settings-modal__danger" onClick={() => setConfirmation("clear_tree")}>Очистить всё дерево</button>
      <small>Карточки будут перемещены в корзину.</small>
    </div>
    <div className="workspace-settings-modal__danger-zone">
      <button type="button" className="workspace-settings-modal__danger" onClick={() => setConfirmation("clear_relations")}>Удалить все связи</button>
      <small>Все браки и родительские связи будут удалены без возможности восстановления.</small>
    </div>
    {error && <div className="workspace-settings-modal__error" role="alert">{error}</div>}
    <div className="workspace-settings-modal__actions"><button type="button" className="workspace-settings-modal__secondary" onClick={onClose}>Назад</button></div>
    {confirmation === "clear_tree" && <ConfirmActionModal title="Очистить всё дерево?" message="Все карточки будут перемещены в корзину." confirmLabel="Очистить дерево" onClose={() => setConfirmation(null)} onConfirm={() => void runAction("clear_tree")} />}
    {confirmation === "clear_relations" && <ConfirmActionModal title="Удалить все связи?" message="Все браки и родительские связи будут удалены без возможности восстановления." confirmLabel="Удалить связи" onClose={() => setConfirmation(null)} onConfirm={() => void runAction("clear_relations")} />}
  </Modal>;
}
