import { GENERATION_HEIGHT } from "@/lib/canvas/generationConstants";
import { usePhotoSrc } from "@/lib/person/usePhotoSrc";
import type { CanvasCardData } from "@/types/canvasCard";
import type { PersonData } from "@/types/person";
import "@/components/PersonList/PersonList.css";

function PersonListAvatar({ photoUrl, sex }: { photoUrl?: string; sex?: PersonData["sex"] }) {
  const src = usePhotoSrc(photoUrl);
  return <span className="person-list__avatar">{src ? <img src={src} alt="" /> : (sex === "female" ? "♀" : sex === "male" ? "♂" : "•")}</span>;
}

type Props = {
  cards: CanvasCardData[];
  roles?: Record<string, string | undefined>;
  onActivate?: (id: string) => void;
  onEdit?: (id: string) => void;
  onDelete?: (id: string) => void;
  showGeneration?: boolean;
};

export function PersonList({ cards, roles = {}, onActivate, onEdit, onDelete, showGeneration = true }: Props) {
  const groups = [...cards.reduce((result, card) => {
    const generation = Math.round(card.y / GENERATION_HEIGHT);
    result.set(generation, [...(result.get(generation) ?? []), card]);
    return result;
  }, new Map<number, CanvasCardData[]>()).entries()];

  return <div className="person-list">
    {groups.map(([generation, generationCards]) => <section key={generation} className="person-list__generation">
      {showGeneration && <h3>Поколение {generation > 0 ? `+${generation}` : generation}</h3>}
      <div className="person-list__items">
        {generationCards.map((card) => {
          const name = [card.person.firstName, card.person.patronymic, card.person.lastName].filter(Boolean).join(" ") || "Без имени";
          const row = <>
            <PersonListAvatar photoUrl={card.person.photoUrl} sex={card.person.sex} />
            <span className="person-list__text"><strong>{name}</strong>{roles[card.id] && <small className="person-list__role">{roles[card.id]}</small>}<small>{card.id}</small></span>
          </>;
          return <div className="person-list__item" key={card.id}>
            {onActivate ? <button type="button" className="person-list__person" onClick={() => onActivate(card.id)}>{row}<span aria-hidden="true">›</span></button> : <div className="person-list__person person-list__person--static">{row}</div>}
            {onEdit && <button type="button" className="person-list__action" onClick={() => onEdit(card.id)} aria-label={`Редактировать ${name}`}>✎</button>}
            {onDelete && <button type="button" className="person-list__action person-list__action--delete" onClick={() => onDelete(card.id)} aria-label={`Удалить ${name}`}>♜</button>}
          </div>;
        })}
      </div>
    </section>)}
  </div>;
}
