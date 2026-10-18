import { useEffect, useState } from "react";
import type { PersonData } from "@/types/person";
import { usePhotoSrc } from "@/lib/person/usePhotoSrc";
import "@/components/PersonCard/PersonCard.css";

type PersonCardProps = {
  person: PersonData;
  selected?: boolean;
};

export function PersonCard({ person, selected = false }: PersonCardProps) {
  const genderClass = person.sex ? ` person-card--${person.sex}` : "";
  const selectedClass = selected ? " person-card--selected" : "";
  const className = `person-card${genderClass}${selectedClass}`;
  const placeholderClass = person.sex
    ? `person-card__photo-placeholder person-card__photo-placeholder--${person.sex}`
    : "person-card__photo-placeholder";
  const placeholderSymbol = person.sex === "male" ? "♂" : person.sex === "female" ? "♀" : "♙";
  const fullName = [person.firstName, person.patronymic]
    .filter(Boolean)
    .join(" ");
  const photoSource = usePhotoSrc(person.photoUrl);
  const [photoFailed, setPhotoFailed] = useState(false);

  useEffect(() => {
    setPhotoFailed(false);
  }, [photoSource]);

  return (
    <article className={className}>
      <div className="person-card__photo">
        {photoSource && !photoFailed ? (
          <img src={photoSource} alt={fullName || "Фото человека"} draggable={false} onError={() => setPhotoFailed(true)} />
        ) : (
          <span className={placeholderClass} aria-label="Фото отсутствует">
            {placeholderSymbol}
          </span>
        )}
        {person.deathDate && <span className="person-card__mourning-ribbon" aria-label="Человек умер" title="Человек умер" />}
      </div>
      <div className="person-card__details">
        {fullName && <div className="person-card__name">{fullName}</div>}
        {person.lastName && <div className="person-card__surname">{person.lastName}</div>}
        {person.previousLastName && (
          <div className="person-card__line">
            <span>{person.sex === "female" ? "Урождённая" : "До изменения"}</span>
            <strong>{person.previousLastName}</strong>
          </div>
        )}
        {person.birthDate && (
          <div className="person-card__line">
            <span>Рождение</span>
            <strong>{person.birthDate}</strong>
          </div>
        )}
        {person.deathDate && (
          <div className="person-card__line">
            <span>Смерть</span>
            <strong>{person.deathDate}</strong>
          </div>
        )}
      </div>
    </article>
  );
}
