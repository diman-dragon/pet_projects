import { useMemo, useState } from "react";
import { Modal } from "@/components/Modal/Modal";
import { ConfirmActionModal } from "@/components/ConfirmActionModal/ConfirmActionModal";
import { PersonList } from "@/components/PersonList/PersonList";
import { useCanvasCardsStore } from "@/store/useCanvasCardsStore";
import { GENERATION_HEIGHT } from "@/lib/canvas/generationConstants";
import "@/components/PeopleListModal/PeopleListModal.css";

type Props = { onClose: () => void };
export function PeopleListModal({ onClose }: Props) {
  const cards = useCanvasCardsStore((state) => state.cards); const removeCard = useCanvasCardsStore((state) => state.removeCard); const [search, setSearch] = useState(""); const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const grouped = useMemo(() => { const query = search.trim().toLocaleLowerCase(); const result = new Map<number, typeof cards>(); cards.filter((card) => [card.person.firstName, card.person.patronymic, card.person.lastName, card.id].filter(Boolean).join(" ").toLocaleLowerCase().includes(query)).sort((a, b) => a.y - b.y || a.x - b.x).forEach((card) => { const generation = Math.round(card.y / GENERATION_HEIGHT); result.set(generation, [...(result.get(generation) ?? []), card]); }); return [...result.entries()]; }, [cards, search]);
  function focus(id: string) { window.dispatchEvent(new CustomEvent("family-tree:focus-card", { detail: { id } })); onClose(); }
  function edit(id: string) { window.dispatchEvent(new CustomEvent("family-tree:edit-card", { detail: { id } })); onClose(); }
  async function deleteCard() { if (!pendingDelete) return; await removeCard(pendingDelete); setPendingDelete(null); }
  return <Modal onClose={onClose} onBack={onClose} backdropClassName="people-list-backdrop" className="people-list-modal" labelledBy="people-list-title">
    <div className="people-list-modal__header"><div><span className="people-list-modal__eyebrow">Дерево</span><h2 id="people-list-title">Список людей</h2></div><button type="button" className="people-list-modal__close" onClick={onClose} aria-label="Закрыть">×</button></div><input className="people-list-modal__search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Поиск по имени или фамилии" autoFocus />
    {!grouped.length ? <div className="people-list-modal__empty">Ничего не найдено</div> : <PersonList cards={grouped.flatMap(([, generationCards]) => generationCards)} onActivate={focus} onEdit={edit} onDelete={setPendingDelete} />}
    {pendingDelete && <ConfirmActionModal title="Удалить карточку?" message="Карточка будет перемещена в корзину вместе со связями." confirmLabel="Удалить" onClose={() => setPendingDelete(null)} onConfirm={() => void deleteCard()} />}
  </Modal>;
}
