type HubCollisionModalProps = {
  onCreateHeight: () => void;
  onMoveHub: () => void;
};

export function HubCollisionModal({ onCreateHeight, onMoveHub }: HubCollisionModalProps) {
  return <Modal onClose={onMoveHub} onBack={onMoveHub} backdropClassName="hub-collision-backdrop" className="hub-collision-modal" labelledBy="hub-collision-title">
    <span className="hub-collision-modal__eyebrow">Пересечение связей</span>
    <h2 id="hub-collision-title">Линии хабов совпали</h2>
    <p>На одной высоте проходят линии двух родительских групп. Выберите, как разнести их.</p>
    <div className="hub-collision-modal__actions"><button type="button" className="hub-collision-modal__secondary" onClick={onMoveHub}>Передвинуть хаб</button><button type="button" className="hub-collision-modal__primary" onClick={onCreateHeight}>Создать новую высоту</button></div>
  </Modal>;
}
import { Modal } from "@/components/Modal/Modal";
