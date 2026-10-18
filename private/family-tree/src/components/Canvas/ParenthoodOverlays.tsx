import { useEffect, useRef, useState, type PointerEvent } from "react";
import { createPortal } from "react-dom";
import type { CanvasCardData } from "@/types/canvasCard";
import type { RelationData } from "@/types/marriage";
import { PERSON_CARD_HEIGHT, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import { MARRIAGE_HEART_SIZE } from "@/lib/canvas/couplePhotoGeometry";
import { getBaseHubY } from "@/lib/canvas/getBaseHubY";
import { getHubConflicts } from "@/lib/canvas/getHubConflicts";
import { getHubLanes } from "@/lib/canvas/getHubLanes";
import { getParenthoodGroups } from "@/lib/canvas/getParenthoodGroups";
import { RelationPath } from "@/components/RelationPath/RelationPath";
import { getMarriageHeartCenter } from "@/lib/canvas/getMarriageHeartCenter";
import { isPointInsideLockedArea } from "@/lib/canvas/lockedAreaStorage";
import { HubCollisionModal } from "@/components/HubCollisionModal/HubCollisionModal";
import "@/components/Canvas/RelationOverlays.css";
import "@/components/Canvas/ParenthoodOverlays.css";

type Point = { x: number; y: number };
type Props = { cards: CanvasCardData[]; relations: RelationData[]; selectedRelationIds?: string[]; onRelationSelect?: (relationIds: string[]) => void; onRelationDoubleClick?: (relationIds: string[]) => void; onRelationContextMenu?: (relationIds: string[], position: { x: number; y: number }) => void };

const LINE_OUTSET = 10;
const HUB_COLLISION_GAP = 8;

export function ParenthoodOverlays({ cards, relations, selectedRelationIds = [], onRelationSelect, onRelationDoubleClick, onRelationContextMenu }: Props) {
  const [hubOffsets, setHubOffsets] = useState<Record<string, number>>({});
  const [createdHubLanes, setCreatedHubLanes] = useState<Record<string, number>>({});
  const [pendingConflict, setPendingConflict] = useState<{ key: string } | null>(null);
  const handledConflicts = useRef(new Set<string>());
  const dragRef = useRef<{ key: string; startY: number; startOffset: number; previousOffset: number } | null>(null);
  const groups = getParenthoodGroups(cards, relations);
  const hubLanes = getHubLanes(groups, createdHubLanes);
  const hubConflicts = getHubConflicts(groups, hubLanes);
  const autoAlignSignature = relations
    .filter((relation) => relation.type === "parenthood")
    .map((relation) => `${relation.id}:${relation.details?.autoAlign === false ? "off" : "on"}`)
    .sort()
    .join("|");

  useEffect(() => {
    setHubOffsets({});
    setCreatedHubLanes({});
    handledConflicts.current.clear();
  }, [autoAlignSignature]);

  useEffect(() => {
    const conflict = hubConflicts.find((item) => !handledConflicts.current.has(item.key));
    if (conflict) setPendingConflict(conflict);
  }, [hubConflicts]);

  function createNewHubHeight() {
    if (!pendingConflict) return;
    const key = pendingConflict.key;
    handledConflicts.current.add(key);
    setCreatedHubLanes((current) => ({ ...current, [key]: (current[key] ?? 0) + 1 }));
    setPendingConflict(null);
  }

  function moveHubManually() {
    if (!pendingConflict) return;
    handledConflicts.current.add(pendingConflict.key);
    setPendingConflict(null);
  }

  function handleHubPointerDown(event: PointerEvent<SVGCircleElement>, key: string, baseY: number) {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { key, startY: event.clientY, startOffset: hubOffsets[key] ?? 0, previousOffset: hubOffsets[key] ?? 0 };
    (event.currentTarget as SVGCircleElement).dataset.baseY = String(baseY);
  }

  function handleHubPointerMove(event: PointerEvent<SVGCircleElement>) {
    const drag = dragRef.current;
    if (!drag || drag.key !== event.currentTarget.dataset.key) return;
    const zoom = Math.max(0.1, Number(event.currentTarget.dataset.zoom ?? "1"));
    const nextOffset = drag.startOffset + (event.clientY - drag.startY) / zoom;
    const group = groups.get(drag.key);
    if (group) {
      const centers = group.children.map((child) => child.x + PERSON_CARD_WIDTH / 2);
      const hubX = (Math.min(...centers) + Math.max(...centers)) / 2;
      const base = getBaseHubY(group).hubY;
      const candidateY = base + (hubLanes.get(drag.key) ?? 0) + nextOffset;
      if (isPointInsideLockedArea(hubX, candidateY)) return;
    }
    setHubOffsets((current) => ({ ...current, [drag.key]: nextOffset }));
  }

  function handleHubPointerUp(event: PointerEvent<SVGCircleElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    event.currentTarget.releasePointerCapture(event.pointerId);
    const candidate = hubOffsets[drag.key] ?? drag.startOffset;
    const base = getBaseHubY(groups.get(drag.key)! ).hubY;
    const candidateY = base + (hubLanes.get(drag.key) ?? 0) + candidate;
    const draggedGroup = groups.get(drag.key)!;
    const draggedCenters = draggedGroup.children.map((child) => child.x + PERSON_CARD_WIDTH / 2);
    const draggedStart = Math.min(...draggedCenters);
    const draggedEnd = Math.max(...draggedCenters);
    const collision = [...groups.keys()].some((key) => {
      if (key === drag.key) return false;
      const otherBase = getBaseHubY(groups.get(key)!).hubY;
      const otherY = otherBase + (hubLanes.get(key) ?? 0) + (hubOffsets[key] ?? 0);
      if (Math.abs(candidateY - otherY) >= HUB_COLLISION_GAP) return false;
      const otherCenters = groups.get(key)!.children.map((child) => child.x + PERSON_CARD_WIDTH / 2);
      const otherStart = Math.min(...otherCenters);
      const otherEnd = Math.max(...otherCenters);
      return draggedStart <= otherEnd && draggedEnd >= otherStart;
    });
    if (collision) setHubOffsets((current) => ({ ...current, [drag.key]: drag.previousOffset }));
    dragRef.current = null;
  }

  return <>
    {pendingConflict && createPortal(<HubCollisionModal onCreateHeight={createNewHubHeight} onMoveHub={moveHubManually} />, document.body)}
    {[...groups.entries()].map(([key, group]) => {
      const { children, parents } = group;
      const base = getBaseHubY(group);
      const childIsBelow = base.childIsBelow;
      const childEdgeY = childIsBelow
        ? Math.min(...children.map((child) => child.y - PERSON_CARD_HEIGHT / 2))
        : Math.max(...children.map((child) => child.y + PERSON_CARD_HEIGHT / 2));
      const baseHubY = base.hubY;
      const hubY = baseHubY + (hubLanes.get(key) ?? 0) + (hubOffsets[key] ?? 0);
      const childCenters = children.map((child) => child.x + PERSON_CARD_WIDTH / 2).sort((first, second) => first - second);
      const horizontalStart = childCenters[0];
      const horizontalEnd = childCenters[childCenters.length - 1];
      const marriage = parents.length === 2 && relations.find((relation) => relation.type === "marriage" &&
        ((relation.personAId === parents[0].id && relation.personBId === parents[1].id) ||
          (relation.personAId === parents[1].id && relation.personBId === parents[0].id)));
      const parenthoodRelationIds = relations.filter((relation) => relation.type === "parenthood" && parents.some((parent) => relation.personAId === parent.id) && children.some((child) => relation.personBId === child.id)).map((relation) => relation.id);
      const marriageHeart = marriage ? getMarriageHeartCenter(marriage, cards) : undefined;
      const parentHub = marriageHeart ? {
        x: marriageHeart.x,
        y: marriageHeart.y,
      } : parents.length === 1 ? {
        x: parents[0].x + PERSON_CARD_WIDTH / 2,
        y: childIsBelow ? parents[0].y + PERSON_CARD_HEIGHT / 2 : parents[0].y - PERSON_CARD_HEIGHT / 2,
      } : null;
      const childHub = { x: (horizontalStart + horizontalEnd) / 2, y: hubY };
      const hubGap = 10;
      const segments = children.map((child) => ({
        childId: child.id,
        start: { x: child.x + PERSON_CARD_WIDTH / 2, y: childIsBelow ? child.y - PERSON_CARD_HEIGHT / 2 - LINE_OUTSET : child.y + PERSON_CARD_HEIGHT / 2 + LINE_OUTSET },
        end: { x: child.x + PERSON_CARD_WIDTH / 2, y: childIsBelow ? hubY + hubGap : hubY - hubGap },
      }));

      const parentSegment = parentHub ? {
        start: marriage && marriageHeart
          ? { x: parentHub.x, y: parentHub.y + (childIsBelow ? MARRIAGE_HEART_SIZE / 2 + LINE_OUTSET : -(MARRIAGE_HEART_SIZE / 2 + LINE_OUTSET)) }
          : { x: parentHub.x, y: parentHub.y + (childIsBelow ? LINE_OUTSET : -LINE_OUTSET) },
        end: { x: childHub.x, y: hubY + (childIsBelow ? -hubGap : hubGap) },
      } : null;
      const parentRoute = parentSegment ? [
        parentSegment.start,
        { x: parentSegment.start.x, y: parentSegment.end.y },
        parentSegment.end,
      ] : undefined;
      const childRoutes = segments.map((segment) => [segment.start, segment.end]);
      const routes = [parentRoute, ...childRoutes].filter((route): route is Point[] => Boolean(route));
      const routePoints = routes.flat();
      const left = Math.min(horizontalStart, parentHub?.x ?? horizontalStart, ...routePoints.map((point) => point.x)) - 12;
      const top = Math.min(hubY, parentHub?.y ?? hubY, ...routePoints.map((point) => point.y)) - 12;
      const right = Math.max(horizontalEnd, parentHub?.x ?? horizontalEnd, ...routePoints.map((point) => point.x)) + 12;
      const width = Math.max(24, right - left);
      const bottom = Math.max(childEdgeY, ...routePoints.map((point) => point.y)) + 12;
      const height = Math.max(24, bottom - top);
      const renderRoute = (route: Point[] | undefined, lineKey: string) => route ? <RelationPath key={lineKey} points={route.map((point) => ({ x: point.x - left, y: point.y - top }))} /> : null;
      const renderHitRoute = (route: Point[] | undefined, lineKey: string) => route ? <svg key={`${lineKey}-hit`} className="parenthood-overlay__hit-svg" style={{ left: 0, top: 0, width, height }} width={width} height={height} aria-hidden="true"><polyline className="parenthood-overlay__hit" points={route.map((point) => `${point.x - left},${point.y - top}`).join(" ")} onClick={(event) => { event.stopPropagation(); onRelationSelect?.(parenthoodRelationIds); }} onDoubleClick={(event) => { event.stopPropagation(); onRelationDoubleClick?.(parenthoodRelationIds); }} onContextMenu={(event) => { event.preventDefault(); event.stopPropagation(); onRelationContextMenu?.(parenthoodRelationIds, { x: event.clientX, y: event.clientY }); }} /></svg> : null;
      const leftHubRoute = [{ x: horizontalStart, y: hubY }, { x: childHub.x - hubGap, y: hubY }];
      const rightHubRoute = [{ x: childHub.x + hubGap, y: hubY }, { x: horizontalEnd, y: hubY }];
      const isSelected = parenthoodRelationIds.some((relationId) => selectedRelationIds.includes(relationId));
      return <div key={key} className={`relation-overlay parenthood-overlay${isSelected ? " is-selected" : ""}`} style={{ left, top, width, height }} aria-label="Родительская связь">
        <svg className="relation-overlay__svg parenthood-overlay__svg" width={width} height={height}>
          <circle
            className="parenthood-overlay__hub"
            data-key={key}
            data-zoom="1"
            cx={childHub.x - left}
            cy={childHub.y - top}
            r="6"
            onPointerDown={(event) => handleHubPointerDown(event, key, baseHubY)}
            onPointerMove={handleHubPointerMove}
            onPointerUp={handleHubPointerUp}
          />
        </svg>
        {renderRoute(parentRoute, `${key}-parent-line`)}
        {renderHitRoute(parentRoute, `${key}-parent-line`)}
        {children.length > 1 && <>
          {renderRoute(leftHubRoute, `${key}-left-line`)}
          {renderRoute(rightHubRoute, `${key}-right-line`)}
          {renderHitRoute(leftHubRoute, `${key}-left-line`)}
          {renderHitRoute(rightHubRoute, `${key}-right-line`)}
        </>}
        {childRoutes.map((route, index) => renderRoute(route, `${key}-${index}-line`))}
        {childRoutes.map((route, index) => renderHitRoute(route, `${key}-${index}-line`))}
      </div>;
    })}
  </>;
}
