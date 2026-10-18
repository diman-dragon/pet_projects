import type { CSSProperties } from "react";
import type { CanvasCardData } from "@/types/canvasCard";
import type { RelationData } from "@/types/marriage";
import { PERSON_CARD_HEIGHT, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import { RelationLine } from "@/components/RelationLine/RelationLine";
import { RelationHeart } from "@/components/RelationHeart/RelationHeart";
import { usePhotoSrc } from "@/lib/person/usePhotoSrc";
import { cardsOverlap } from "@/lib/canvas/cardsOverlap";
import { getMarriageEdgePoints } from "@/lib/canvas/getMarriageEdgePoints";
import { getPathMidpoint } from "@/lib/canvas/getPathMidpoint";
import { getSafeMarriageRoute } from "@/lib/canvas/getSafeMarriageRoute";
import type { CanvasPoint as Point } from "@/lib/canvas/segmentIntersectsRect";
import { getCouplePhotoLayout, MARRIAGE_HEART_SIZE } from "@/lib/canvas/couplePhotoGeometry";
import "@/components/Canvas/RelationOverlays.css";
import "@/components/Canvas/MarriageOverlays.css";

type Props = { cards: CanvasCardData[]; relations: RelationData[]; selectedRelationIds?: string[]; onRelationSelect?: (relationId: string) => void; onRelationDoubleClick?: (relationId: string) => void; onRelationContextMenu?: (relationId: string, position: { x: number; y: number }) => void; onCouplePhotoClick?: (relationId: string) => void };

function CouplePhoto({ url, style }: { url: string; style: CSSProperties }) {
  const src = usePhotoSrc(url);
  if (!src) return null;
  return <div className="marriage-overlay__couple-photo" style={style}><img src={src} alt="Фото пары" /></div>;
}
const HEART_SIZE = MARRIAGE_HEART_SIZE;
const ROUTE_CLEARANCE = 12;

function isLaterMarriage(current: RelationData, candidate: RelationData, relations: RelationData[]) {
  const currentDate = String(current.details?.registrationDate ?? "");
  const candidateDate = String(candidate.details?.registrationDate ?? "");
  if (currentDate && candidateDate && candidateDate !== currentDate) return candidateDate > currentDate;
  if (!currentDate && candidateDate) return true;
  return relations.indexOf(candidate) > relations.indexOf(current);
}

function isWidowedPreviousMarriage(relation: RelationData, relations: RelationData[], cards: CanvasCardData[]) {
  const members = [relation.personAId, relation.personBId];
  const hasDeceasedSpouse = members.some((id) => Boolean(cards.find((card) => card.id === id)?.person.deathDate));
  if (!hasDeceasedSpouse) return false;
  return relations.some((candidate) => candidate.id !== relation.id
    && candidate.details?.status !== "dissolved"
    && (members.includes(candidate.personAId) || members.includes(candidate.personBId))
    && isLaterMarriage(relation, candidate, relations));
}

export function MarriageOverlays({ cards, relations, selectedRelationIds = [], onRelationSelect, onRelationDoubleClick, onRelationContextMenu, onCouplePhotoClick }: Props) {
  const marriages = relations.filter((relation) => relation.type === "marriage");
  return <>
    {marriages.map((relation) => {
      if (relation.type !== "marriage") return null;
      const first = cards.find((card) => card.id === relation.personAId);
      const second = cards.find((card) => card.id === relation.personBId);
      if (!first || !second) return null;
      if (cardsOverlap(first, second)) return null;
      const [start, end] = getMarriageEdgePoints(first, second);
      const obstacles = cards.filter((card) => card.id !== first.id && card.id !== second.id).map((card) => ({ left: card.x - ROUTE_CLEARANCE, right: card.x + PERSON_CARD_WIDTH + ROUTE_CLEARANCE, top: card.y - PERSON_CARD_HEIGHT / 2 - ROUTE_CLEARANCE, bottom: card.y + PERSON_CARD_HEIGHT / 2 + ROUTE_CLEARANCE }));
      const path = getSafeMarriageRoute(start, end, obstacles);
      if (!path) return null;
      const left = Math.min(...path.map((point) => point.x), first.x) - 30;
      const top = Math.min(...path.map((point) => point.y), first.y - PERSON_CARD_HEIGHT / 2) - 30;
      const right = Math.max(...path.map((point) => point.x), first.x + PERSON_CARD_WIDTH, second.x + PERSON_CARD_WIDTH) + 30;
      const bottom = Math.max(...path.map((point) => point.y), second.y + PERSON_CARD_HEIGHT / 2) + 30;
      const localPath = path.map((point) => ({ x: point.x - left, y: point.y - top }));
      const localStart = localPath[0];
      const localEnd = localPath[localPath.length - 1];
      const pathMidpoint = getPathMidpoint(localPath);
      const midpoint = { x: pathMidpoint.x, y: pathMidpoint.y };
      const worldMidpoint = { x: midpoint.x + left, y: midpoint.y + top };
      const length = Math.hypot(localEnd.x - localStart.x, localEnd.y - localStart.y) || 1;
      const ux = (localEnd.x - localStart.x) / length;
      const uy = (localEnd.y - localStart.y) / length;
      const gap = HEART_SIZE / 2 + ROUTE_CLEARANCE;
      const before = { x: midpoint.x - ux * gap, y: midpoint.y - uy * gap };
      const after = { x: midpoint.x + ux * gap, y: midpoint.y + uy * gap };
      const status = relation.details?.status;
      const isWidowedPrevious = isWidowedPreviousMarriage(relation, marriages, cards);
      const couplePhotoUrl = typeof relation.details?.couplePhotoUrl === "string" ? relation.details.couplePhotoUrl : undefined;
      const couplePhotoLayout = getCouplePhotoLayout(worldMidpoint.y, first.y);
      const couplePhotoSize = couplePhotoLayout.size;
      const couplePhotoCenter = { x: midpoint.x, y: couplePhotoLayout.top - top + couplePhotoSize / 2 };
      const photoTop = couplePhotoCenter.y - couplePhotoSize / 2;
      const overlayTop = Math.min(top, photoTop + top) - 30;
      const overlayBottom = Math.max(bottom, top + couplePhotoCenter.y + couplePhotoSize / 2) + 30;
      const overlayLeft = Math.min(left, left + couplePhotoCenter.x - couplePhotoSize / 2) - 30;
      const overlayRight = Math.max(right, left + couplePhotoCenter.x + couplePhotoSize / 2) + 30;
      const width = overlayRight - overlayLeft;
      const height = overlayBottom - overlayTop;
      const offsetX = left - overlayLeft;
      const offsetY = top - overlayTop;
      const shiftedStart = { x: localStart.x + offsetX, y: localStart.y + offsetY };
      const shiftedEnd = { x: localEnd.x + offsetX, y: localEnd.y + offsetY };
      const shiftedBefore = { x: before.x + offsetX, y: before.y + offsetY };
      const shiftedAfter = { x: after.x + offsetX, y: after.y + offsetY };
      const renderSegment = (segmentStart: Point, segmentEnd: Point, key: string) => {
        return <RelationLine key={key} start={segmentStart} end={segmentEnd} />;
      };
      return <div key={relation.id} className={`relation-overlay marriage-overlay${selectedRelationIds.includes(relation.id) ? " is-selected" : ""}`} style={{ left: overlayLeft, top: overlayTop, width, height }} onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); onRelationSelect?.(relation.id); }} onDoubleClick={(event) => { event.stopPropagation(); onRelationDoubleClick?.(relation.id); }} onContextMenu={(event) => { event.preventDefault(); event.stopPropagation(); onRelationContextMenu?.(relation.id, { x: event.clientX, y: event.clientY }); }} aria-label="Брачная связь" role="button" tabIndex={0}>
        {path.length === 2 ? <>
          {renderSegment(shiftedStart, shiftedBefore, `${relation.id}-first`)}
          {renderSegment(shiftedAfter, shiftedEnd, `${relation.id}-second`)}
        </> : localPath.slice(0, -1).map((point, index) => renderSegment({ x: point.x + offsetX, y: point.y + offsetY }, { x: localPath[index + 1].x + offsetX, y: localPath[index + 1].y + offsetY }, `${relation.id}-route-${index}`))}
          {couplePhotoUrl && couplePhotoSize > 0 && <CouplePhoto url={couplePhotoUrl} style={{ left: couplePhotoCenter.x + offsetX, top: couplePhotoCenter.y + offsetY, width: couplePhotoSize, height: couplePhotoSize }} />}
          <RelationHeart
            variant={!isWidowedPrevious && status === "dissolved" ? "broken" : "solid"}
          size={HEART_SIZE}
          className={`marriage-overlay__heart${isWidowedPrevious ? " is-widowed" : ""}`}
          style={{ left: midpoint.x + offsetX, top: midpoint.y + offsetY }}
          onClick={(event) => { event.stopPropagation(); onCouplePhotoClick?.(relation.id); }}
        />
      </div>;
    })}
  </>;
}
