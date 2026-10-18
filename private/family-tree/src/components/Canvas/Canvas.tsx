import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { MouseEvent } from "react";
import { useCanvasCardsStore } from "@/store/useCanvasCardsStore";
import { getGenerationCenterY } from "@/lib/canvas/getGenerationCenterY";
import { GENERATION_HEIGHT, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import { ContextMenu } from "@/components/ContextMenu/ContextMenu";
import { getCardsBounds } from "@/lib/canvas/getCardsBounds";
import { getLockedAreas } from "@/lib/canvas/getLockedAreas";
import { saveLockedArea } from "@/lib/canvas/lockedAreaStorage";
import { getParenthoodAlignment } from "@/lib/canvas/getParenthoodAlignment";
import { getActiveMarriageMemberIds, getChildrenForParents, getParenthoodGroupKey, getParentRoleForCard } from "@/lib/canvas/parenthoodRealignment";
import type { ContextMenuPosition, ContextMenuState } from "@/types/contextMenu";
import { CanvasCard } from "@/components/Canvas/CanvasCard";
import { CanvasControls } from "@/components/Canvas/CanvasControls";
import { GenerationGuides } from "@/components/Canvas/GenerationGuides";
import { MarriageOverlays } from "@/components/Canvas/MarriageOverlays";
import { ParenthoodOverlays } from "@/components/Canvas/ParenthoodOverlays";
import { ConfirmActionModal } from "@/components/ConfirmActionModal/ConfirmActionModal";
import { PersonShortModal } from "@/components/PersonShortModal/PersonShortModal";
import { PersonExtendedModal } from "@/components/PersonExtendedModal/PersonExtendedModal";
import { RelationShortModal } from "@/components/RelationShortModal/RelationShortModal";
import { RelationExtendedModal } from "@/components/RelationExtendedModal/RelationExtendedModal";
import { CouplePhotoModal } from "@/components/CouplePhotoModal/CouplePhotoModal";
import { useCanvasViewport } from "@/components/Canvas/useCanvasViewport";
import { useCanvasAreaSelection } from "@/components/Canvas/useCanvasAreaSelection";
import { useCanvasContextMenus } from "@/components/Canvas/useCanvasContextMenus";
import "@/components/Canvas/Canvas.css";

type LockedAreaPreview = { left: number; top: number; width: number; height: number };
type RelationModalState = { type: "marriage" | "parenthood"; personIds: string[]; parentIds?: string[]; pairs: Array<[string, string]>; relationIds: string[]; isNew: boolean; extended?: boolean; autoAlign?: boolean; status: "active" | "dissolved" };

export function Canvas({ showAuxiliaryLines = true }: { showAuxiliaryLines?: boolean }) {
  const {
    canvasRef,
    viewport,
    transform,
    consumePanClick,
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    handleWheel,
    centerView,
    fitViewToBounds,
    screenToWorld,
    zoomIn,
    zoomOut,
    zoomTo100,
  } = useCanvasViewport();
  const [lockedAreaPreview, setLockedAreaPreview] = useState<LockedAreaPreview | null>(null);
  const [personModal, setPersonModal] = useState<{ id?: string; isNew: boolean; extended?: boolean; position?: { x: number; y: number } } | null>(null);
  const [relationModal, setRelationModal] = useState<RelationModalState | null>(null);
  const [couplePhotoModal, setCouplePhotoModal] = useState<{ relationId: string; personId: string; currentUrl?: string } | null>(null);
  const [selectedRelationIds, setSelectedRelationIds] = useState<string[]>([]);
  const [selectionActionPosition, setSelectionActionPosition] = useState<{ x: number; y: number } | null>(null);
  const [selectionPendingPosition, setSelectionPendingPosition] = useState<{ x: number; y: number } | null>(null);
  const { contextMenu, openCanvasMenu, openCardMenu, openRelationMenu, closeContextMenu } = useCanvasContextMenus();
  const cards = useCanvasCardsStore((state) => state.cards);
  const relations = useCanvasCardsStore((state) => state.relations);
  const loadGraph = useCanvasCardsStore((state) => state.loadGraph);
  const addCard = useCanvasCardsStore((state) => state.addCard);
  const removeCard = useCanvasCardsStore((state) => state.removeCard);
  const createRelation = useCanvasCardsStore((state) => state.createRelation);
  const updateRelation = useCanvasCardsStore((state) => state.updateRelation);
  const updateParenthoodAutoAlign = useCanvasCardsStore((state) => state.updateParenthoodAutoAlign);
  const deleteRelation = useCanvasCardsStore((state) => state.deleteRelation);
  const updatePerson = useCanvasCardsStore((state) => state.updatePerson);
  const selectCard = useCanvasCardsStore((state) => state.selectCard);
  const toggleCardSelection = useCanvasCardsStore((state) => state.toggleCardSelection);
  const selectedCardIds = useCanvasCardsStore((state) => state.selectedCardIds);
  const lockedAreas = getLockedAreas(cards);

  const canUndo = useCanvasCardsStore((state) => state.canUndo);
  const undoLastAction = useCanvasCardsStore((state) => state.undoLastAction);
  const alignCardsToCenter = useCanvasCardsStore((state) => state.alignCardsToCenter);
  const setCardsLocked = useCanvasCardsStore((state) => state.setCardsLocked);
  const [confirmation, setConfirmation] = useState<{ title: string; message: string; confirmLabel?: string; action: () => void } | null>(null);
  const { areaSelection, handlePointerDown: handleCanvasPointerDown, handlePointerMove: handleCanvasPointerMove, handlePointerUp: handleCanvasPointerUp, clearAreaSelection, consumeSelectionClick } = useCanvasAreaSelection({
    canvasRef,
    cards,
    screenToWorld,
    onPanPointerDown: handlePointerDown,
    onPanPointerMove: handlePointerMove,
    onPanPointerUp: handlePointerUp,
  });

  useEffect(() => {
    function closeCanvasOverlays(event: KeyboardEvent) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        const activeElement = document.activeElement;
        const isInput = activeElement?.tagName === "INPUT" || activeElement?.tagName === "TEXTAREA" || activeElement?.getAttribute("contenteditable") === "true";
        const isModalOpen = personModal !== null || relationModal !== null || couplePhotoModal !== null || confirmation !== null;
        if (!isInput && !isModalOpen) {
          event.preventDefault();
          if (useCanvasCardsStore.getState().canUndo) void useCanvasCardsStore.getState().undoLastAction();
        }
        return;
      }
      if (event.key !== "Escape") return;
      closeContextMenu();
      clearAreaSelection();
      setConfirmation(null);
      setRelationModal(null);
      setSelectedRelationIds([]);
      setSelectionActionPosition(null);
      setSelectionPendingPosition(null);
      selectCard(null);
    }
    document.addEventListener("keydown", closeCanvasOverlays);
    return () => document.removeEventListener("keydown", closeCanvasOverlays);
  }, [clearAreaSelection, closeContextMenu, selectCard]);

  useEffect(() => {
    void loadGraph();
  }, [loadGraph]);

  useEffect(() => {
    function handleFocusCard(event: Event) {
      const id = (event as CustomEvent<{ id?: string }>).detail?.id;
      const card = id ? useCanvasCardsStore.getState().cards.find((item) => item.id === id) : undefined;
      if (!card) return;
      selectCard(card.id);
      centerView({ x: card.x + PERSON_CARD_WIDTH / 2, y: card.y });
    }
    window.addEventListener("family-tree:focus-card", handleFocusCard);
    return () => window.removeEventListener("family-tree:focus-card", handleFocusCard);
  }, [centerView, selectCard]);

  useEffect(() => {
    function handleEditCard(event: Event) {
      const id = (event as CustomEvent<{ id?: string }>).detail?.id;
      if (id && useCanvasCardsStore.getState().cards.some((card) => card.id === id)) setPersonModal({ id, isNew: false });
    }
    window.addEventListener("family-tree:edit-card", handleEditCard);
    return () => window.removeEventListener("family-tree:edit-card", handleEditCard);
  }, []);

  useEffect(() => {
    function handleWorkspaceChanged() {
      void loadGraph();
    }
    window.addEventListener("family-tree:workspace-changed", handleWorkspaceChanged);
    return () => window.removeEventListener("family-tree:workspace-changed", handleWorkspaceChanged);
  }, [loadGraph]);

  function getRelationSelection(cardIds: string[]): RelationModalState | null {
    if (cardIds.length < 2) return null;
    const selected = cardIds.map((id) => cards.find((card) => card.id === id)).filter((card): card is typeof cards[number] => Boolean(card));
    if (selected.length !== cardIds.length) return null;
    const generations = [...new Set(selected.map((card) => Math.round(card.y / GENERATION_HEIGHT)))];
    if (generations.length === 1 && selected.length === 2) {
      const existing = relations.find((relation) => relation.type === "marriage" && cardIds.includes(relation.personAId) && cardIds.includes(relation.personBId));
      const hasDeceasedPreviousSpouse = selected.some((card) =>
        !card.person.deathDate && relations.some((relation) => {
          if (relation.type !== "marriage" || (relation.personAId !== card.id && relation.personBId !== card.id)) return false;
          const spouseId = relation.personAId === card.id ? relation.personBId : relation.personAId;
          return Boolean(cards.find((candidate) => candidate.id === spouseId)?.person.deathDate);
        }),
      );
      return { type: "marriage", personIds: cardIds, pairs: [[cardIds[0], cardIds[1]]], relationIds: existing ? [existing.id] : [], isNew: !existing, status: existing ? (existing.details?.status === "dissolved" ? "dissolved" : "active") : (hasDeceasedPreviousSpouse ? "active" : "dissolved") };
    }
    if (generations.length !== 2 || Math.max(...generations) - Math.min(...generations) !== 1) return null;
    const parents = selected.filter((card) => Math.round(card.y / GENERATION_HEIGHT) === Math.min(...generations));
    const children = selected.filter((card) => Math.round(card.y / GENERATION_HEIGHT) !== Math.min(...generations));
    if (parents.length < 1 || parents.length > 2 || !children.length) return null;
    const pairs = parents.flatMap((parent) => children.map((child) => [parent.id, child.id] as [string, string]));
    const existing = relations.filter((relation) => relation.type === "parenthood" && pairs.some(([parentId, childId]) => relation.personAId === parentId && relation.personBId === childId));
    return { type: "parenthood", personIds: cardIds, parentIds: parents.map((parent) => parent.id), pairs, relationIds: existing.map((relation) => relation.id), isNew: existing.length === 0, autoAlign: existing.every((relation) => relation.details?.autoAlign !== false), status: "active" };
  }

  useEffect(() => {
    if (selectedCardIds.length < 2) setSelectionActionPosition(null);
    if (selectedCardIds.length > 0) setSelectedRelationIds([]);
  }, [selectedCardIds.length]);

  useEffect(() => {
    function handleKeyUp(event: KeyboardEvent) {
      if (event.key !== "Control" && event.key !== "Meta") return;
      const position = selectionPendingPosition;
      if (position && getRelationSelection(useCanvasCardsStore.getState().selectedCardIds)) setSelectionActionPosition(position);
      else setSelectionActionPosition(null);
      setSelectionPendingPosition(null);
    }
    function handleWindowBlur() {
      setSelectionPendingPosition(null);
      setSelectionActionPosition(null);
    }
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", handleWindowBlur);
    return () => { window.removeEventListener("keyup", handleKeyUp); window.removeEventListener("blur", handleWindowBlur); };
  }, [cards, relations, selectionPendingPosition]);


  useEffect(() => {
    function handleLockedAreaFocus(event: Event) {
      const cardIds = (event as CustomEvent<{ cardIds?: string[] }>).detail?.cardIds ?? [];
      const storedArea = getLockedAreas(useCanvasCardsStore.getState().cards).find((area) => area.cards.length === cardIds.length && area.cards.every((card) => cardIds.includes(card.id)));
      const bounds = storedArea?.bounds;
      if (!bounds) return;
      const padding = 48;
      const preview = { left: bounds.left - padding, top: bounds.top - padding, width: bounds.right - bounds.left + padding * 2, height: bounds.bottom - bounds.top + padding * 2 };
      setLockedAreaPreview(preview);
      fitViewToBounds({ left: preview.left, right: preview.left + preview.width, top: preview.top, bottom: preview.top + preview.height });
    }
    function handleLockedAreaClear() {
      setLockedAreaPreview(null);
    }
    window.addEventListener("family-tree:focus-locked-area", handleLockedAreaFocus);
    window.addEventListener("family-tree:clear-locked-area-preview", handleLockedAreaClear);
    return () => {
      window.removeEventListener("family-tree:focus-locked-area", handleLockedAreaFocus);
      window.removeEventListener("family-tree:clear-locked-area-preview", handleLockedAreaClear);
    };
  }, [fitViewToBounds]);

  function handleContextMenu(event: MouseEvent<HTMLDivElement>) {
    event.preventDefault();
    openCanvasMenu({ x: event.clientX, y: event.clientY });
  }

  function handleBackgroundClick() {
    if (consumePanClick()) return;
    if (consumeSelectionClick()) return;
    closeContextMenu();
    setSelectionActionPosition(null);
    setSelectionPendingPosition(null);
    setSelectedRelationIds([]);
    if (areaSelection) clearAreaSelection();
    selectCard(null);
  }

  function handleSelectionToggle(cardId: string, clientX: number, clientY: number) {
    const ids = useCanvasCardsStore.getState().selectedCardIds;
    if (ids.length < 2) {
      setSelectionActionPosition(null);
      setSelectionPendingPosition({ x: clientX, y: clientY });
      return;
    }
    if (!getRelationSelection(ids)) {
      toggleCardSelection(cardId);
      setSelectionActionPosition(null);
      setSelectionPendingPosition(null);
      return;
    }
    setSelectionActionPosition(null);
    setSelectionPendingPosition({ x: clientX, y: clientY });
  }

  function openRelationModalFromSelection() {
    const selection = getRelationSelection(useCanvasCardsStore.getState().selectedCardIds);
    if (!selection) return;
    setRelationModal(selection);
    setSelectionActionPosition(null);
  }

  function openRelationModalForRelation(relationIds: string[]) {
    const selectedRelations = relationIds.map((id) => relations.find((relation) => relation.id === id)).filter((relation): relation is typeof relations[number] => Boolean(relation));
    const relation = selectedRelations[0];
    if (!relation) return;
    if (relation.type === "marriage") {
      const personIds = [relation.personAId, relation.personBId];
      setRelationModal({ type: "marriage", personIds, pairs: [[relation.personAId, relation.personBId]], relationIds: [relation.id], isNew: false, status: relation.details?.status === "dissolved" ? "dissolved" : "active" });
      return;
    }
    const parentIds = [...new Set(selectedRelations.filter((item) => item.type === "parenthood").map((item) => item.personAId))];
    const childIds = [...new Set(selectedRelations.filter((item) => item.type === "parenthood").map((item) => item.personBId))];
    setRelationModal({ type: "parenthood", personIds: [...parentIds, ...childIds], parentIds, pairs: selectedRelations.filter((item) => item.type === "parenthood").map((item) => [item.personAId, item.personBId]), relationIds: selectedRelations.map((item) => item.id), isNew: false, autoAlign: selectedRelations.every((item) => item.details?.autoAlign !== false), status: "active" });
  }

  function selectRelation(relationIds: string[]) {
    setSelectedRelationIds(relationIds);
    selectCard(null);
  }

  function openCouplePhotoModal(relationId: string) {
    const relation = relations.find((item) => item.id === relationId && item.type === "marriage");
    if (!relation) return;
    setCouplePhotoModal({ relationId, personId: relation.personAId, currentUrl: typeof relation.details?.couplePhotoUrl === "string" ? relation.details.couplePhotoUrl : undefined });
  }

  async function saveCouplePhoto(url: string) {
    if (!couplePhotoModal) return;
    const relation = relations.find((item) => item.id === couplePhotoModal.relationId);
    if (!relation) return;
    await updateRelation({ relationId: relation.id, relationType: "marriage", personAId: relation.personAId, personBId: relation.personBId, detailsJson: JSON.stringify({ ...(relation.details ?? {}), couplePhotoUrl: url }), allowExistingMarriage: true });
    setCouplePhotoModal(null);
  }

  async function handleAddCard(position: ContextMenuPosition) {
    const point = screenToWorld(position);
    if (point) {
      setPersonModal({ isNew: true, position: { x: point.x, y: getGenerationCenterY(point.y) } });
    }
    closeContextMenu();
  }

  function handleCardContextMenu(cardId: string, position: ContextMenuPosition) {
    selectCard(cardId);
    openCardMenu(cardId, position);
  }

  function handleEditCard(cardId: string) { selectCard(cardId); closeContextMenu(); setPersonModal({ id: cardId, isNew: false }); }

  async function savePerson(person: NonNullable<typeof cards[number]["person"]>, photoBytes?: number[]) {
    if (!personModal) return;
    let cardId = personModal.id;
    if (personModal.isNew && personModal.position) {
      cardId = await addCard(personModal.position.x, personModal.position.y);
    }
    if (!cardId) return;
    let nextPerson = person;
    if (photoBytes?.length) {
      const { invoke } = await import("@tauri-apps/api/core");
      const photoUrl = await invoke<string>("save_person_photo_crop", {
        personId: cardId,
        imageData: photoBytes,
        zoom: 1,
        offsetX: 0,
        offsetY: 0,
      });
      nextPerson = { ...person, photoUrl, photoZoom: 1, photoOffsetX: 0, photoOffsetY: 0 };
    }
    await updatePerson(cardId, nextPerson);
    await loadGraph();
    if (personModal.isNew) {
      selectCard(cardId);
      setPersonModal(null);
      return;
    }
    setPersonModal((current) => current?.id === cardId ? { ...current, isNew: false } : current);
  }

  function closePersonModal() { setPersonModal(null); }

  async function togglePersonLock(cardId: string, locked: boolean) {
    await setCardsLocked([cardId], !locked);
  }

  async function saveRelation(status: "active" | "dissolved") {
    if (!relationModal) return;
    const relationIds: string[] = [];
    for (let index = 0; index < relationModal.pairs.length; index += 1) {
      const [personAId, personBId] = relationModal.pairs[index];
      const existingRelation = relations.find((relation) => relation.type === relationModal.type && ((relation.personAId === personAId && relation.personBId === personBId) || (relationModal.type === "marriage" && relation.personAId === personBId && relation.personBId === personAId)));
      const relationId = relationModal.relationIds[index] ?? existingRelation?.id;
      const previousDetails = relationId ? relations.find((relation) => relation.id === relationId)?.details ?? {} : {};
      const details = relationModal.type === "marriage" ? { ...previousDetails, status } : { ...previousDetails, role: cards.find((card) => card.id === personAId)?.person.sex === "female" ? "mother" : "father", autoAlign: relationModal.autoAlign !== false };
      if (relationId) {
        if (!relationModal.isNew) {
          await updateRelation({ relationId, relationType: relationModal.type, personAId, personBId, detailsJson: JSON.stringify(details), allowExistingMarriage: true });
        }
        relationIds.push(relationId);
      } else {
        relationIds.push(await createRelation({ relationType: relationModal.type, personAId, personBId, detailsJson: JSON.stringify(details), allowExistingMarriage: false }));
      }
    }
    await loadGraph();
    if (relationModal.isNew) {
      setRelationModal(null);
      selectCard(null);
    } else {
      setRelationModal((current) => current ? { ...current, relationIds, isNew: false, status } : current);
    }
  }

  async function setParenthoodAutoAlign(autoAlign: boolean) {
    if (!relationModal || relationModal.type !== "parenthood") return;
    setRelationModal((current) => current ? { ...current, autoAlign } : current);
    if (relationModal.isNew || !relationModal.parentIds?.length) return;
    const parentIds = [...relationModal.parentIds].sort();
    const familyRelationIds = relations.filter((relation) => {
      if (relation.type !== "parenthood") return false;
      const childParentIds = [...new Set(relations.filter((candidate) => candidate.type === "parenthood" && candidate.personBId === relation.personBId).map((candidate) => candidate.personAId))].sort();
      return childParentIds.length === parentIds.length && childParentIds.every((id, index) => id === parentIds[index]);
    }).map((relation) => relation.id);
    await updateParenthoodAutoAlign(familyRelationIds, autoAlign);
    if (autoAlign) {
      const updatedRelations = useCanvasCardsStore.getState().relations;
      const childIds = [...new Set(familyRelationIds.map((relationId) => updatedRelations.find((relation) => relation.id === relationId)?.personBId).filter((id): id is string => Boolean(id)))];
      const processedGroups = new Set<string>();
      for (const childId of childIds) {
        const parentRelation = updatedRelations.find((relation) => relation.type === "parenthood" && relation.personBId === childId);
        if (!parentRelation) continue;
        const state = useCanvasCardsStore.getState();
        const groupKey = getParenthoodGroupKey(state.cards, updatedRelations, childId);
        if (processedGroups.has(groupKey)) continue;
        processedGroups.add(groupKey);
        const role = getParentRoleForCard(updatedRelations, childId, parentRelation.personAId);
        await alignChildrenUnderParents(childId, role, parentRelation.personAId, true, true);
      }
    }
  }

  async function saveRelationDetails(details: Record<string, unknown>) {
    if (!relationModal || !relationModal.relationIds[0]) return;
    const relation = relations.find((item) => item.id === relationModal.relationIds[0]);
    if (!relation) return;
    await updateRelation({ relationId: relation.id, relationType: relation.type, personAId: relation.personAId, personBId: relation.personBId, detailsJson: JSON.stringify(details), allowExistingMarriage: true });
    await loadGraph();
    setRelationModal((current) => current ? { ...current, extended: true, status: details.status === "dissolved" ? "dissolved" : "active" } : current);
  }

  async function applyDeleteRelations(relationIds: string[]) {
    for (const relationId of relationIds) await deleteRelation(relationId);
    await loadGraph();
    setRelationModal(null);
    setConfirmation(null);
    closeContextMenu();
    selectCard(null);
  }

  function requestDeleteRelations(relationIds: string[]) {
    setConfirmation({ title: "Удалить связь?", message: `Будет удалено связей: ${relationIds.length}. Действие можно отменить через undo.`, confirmLabel: "Удалить", action: () => { void applyDeleteRelations(relationIds); } });
  }

  async function applyDeleteCard(cardId: string) {
    await removeCard(cardId);
    setPersonModal(null);
    setRelationModal(null);
    setSelectionActionPosition(null);
    setSelectionPendingPosition(null);
    selectCard(null);
    closeContextMenu();
  }

  function handleDeleteCard(cardId: string) {
    if (cards.find((card) => card.id === cardId)?.locked) return;
    setConfirmation({ title: "Удалить карточку?", message: "Карточка будет перемещена в корзину. Её можно будет восстановить.", confirmLabel: "Удалить", action: () => { void applyDeleteCard(cardId); } });
  }

  async function alignChildrenUnderParents(childId: string, role: "father" | "mother", changedParentId: string, preserveGeneration = false, force = false, includeParentsInGroup = false) {
    await loadGraph();
    const parentRelations = useCanvasCardsStore.getState().relations.filter((relation) => relation.type === "parenthood" && relation.personBId === childId);
    if (!force && parentRelations.some((relation) => relation.details?.autoAlign === false)) return;
    const state = useCanvasCardsStore.getState();
    const alignment = getParenthoodAlignment(state.cards, state.relations, childId, role, changedParentId, preserveGeneration, includeParentsInGroup);
    if (alignment.requiresMarriage) {
      return;
    }
    if (alignment.targetCenter !== undefined && alignment.targetGenerationY !== undefined) await alignCardsToCenter(alignment.groupIds, alignment.targetCenter, alignment.targetGenerationY, alignment.children);
    else if (alignment.targetCenter !== undefined) await alignCardsToCenter(alignment.groupIds, alignment.targetCenter, undefined, alignment.children);
  }

  async function alignParentsOverChildren(childId: string) {
    await loadGraph();
    const state = useCanvasCardsStore.getState();
    const parentRelations = state.relations.filter((relation) => relation.type === "parenthood" && relation.personBId === childId);
    if (!parentRelations.length || parentRelations.some((relation) => relation.details?.autoAlign === false)) return;
    const parentIds = [...new Set(parentRelations.map((relation) => relation.personAId))];
    const childCard = state.cards.find((card) => card.id === childId);
    if (!childCard) return;
    const childGeneration = getGenerationCenterY(childCard.y);
    const children = state.relations
      .filter((relation) => relation.type === "parenthood")
      .map((relation) => relation.personBId)
      .filter((candidateId, index, ids) => ids.indexOf(candidateId) === index)
      .map((candidateId) => {
        const candidateRelations = state.relations.filter((relation) => relation.type === "parenthood" && relation.personBId === candidateId);
        const candidateParents = [...new Set(candidateRelations.map((relation) => relation.personAId))].sort();
        const targetParents = [...parentIds].sort();
        const candidate = state.cards.find((card) => card.id === candidateId);
        return candidate && candidateParents.length === targetParents.length && candidateParents.every((id, parentIndex) => id === targetParents[parentIndex]) && getGenerationCenterY(candidate.y) === childGeneration ? candidate : undefined;
      })
      .filter((card): card is typeof state.cards[number] => Boolean(card));
    if (!children.length) return;
    const targetCenter = (Math.min(...children.map((card) => card.x)) + Math.max(...children.map((card) => card.x + PERSON_CARD_WIDTH))) / 2;
    const groupIds = [...new Set(parentIds.flatMap((parentId) => [...getActiveMarriageMemberIds(state.relations, parentId)]))];
    await alignCardsToCenter(groupIds, targetCenter, undefined, parentIds);
  }

  async function centerRelation(relationIds: string[]) {
    const relation = relations.find((item) => relationIds.includes(item.id));
    if (!relation || relation.type !== "parenthood") return;
    const role = getParentRoleForCard(relations, relation.personBId, relation.personAId);
    await alignChildrenUnderParents(relation.personBId, role, relation.personAId, true, true);
    closeContextMenu();
  }

  async function realignParentGroupsForCard(cardId: string) {
    await loadGraph();
    const currentCards = useCanvasCardsStore.getState().cards;
    const currentRelations = useCanvasCardsStore.getState().relations;
    const movedIds = getActiveMarriageMemberIds(currentRelations, cardId);
    const children = new Set<string>([
      ...getChildrenForParents(currentRelations, movedIds),
      ...currentRelations.filter((relation) => relation.type === "parenthood" && movedIds.has(relation.personBId)).map((relation) => relation.personBId),
    ]);
    const processedGroups = new Set<string>();
    for (const childId of children) {
      const parentRelation = currentRelations.find((relation) => relation.type === "parenthood" && relation.personBId === childId);
      if (!parentRelation) continue;
      const groupKey = getParenthoodGroupKey(currentCards, currentRelations, childId);
      if (processedGroups.has(groupKey)) continue;
      processedGroups.add(groupKey);
      const changedParentId = [...movedIds].find((id) => currentRelations.some((relation) => relation.type === "parenthood" && relation.personBId === childId && relation.personAId === id)) ?? parentRelation.personAId;
      const movedChild = movedIds.has(childId);
      if (movedChild) {
        await alignParentsOverChildren(childId);
      } else {
        const role = getParentRoleForCard(currentRelations, childId, changedParentId);
        await alignChildrenUnderParents(childId, role, changedParentId, true);
      }
    }
  }

  function handleFitTree() {
    fitViewToBounds(getCardsBounds(cards));
    selectCard(null);
  }

  async function handleUndo() {
    closeContextMenu();
    await undoLastAction();
  }

  function getContextMenuItems() {
    if (!contextMenu) return [];

    if (contextMenu.type === "canvas") {
      return [
        ...(canUndo ? [{ label: "Отменить действие", onSelect: () => void handleUndo() }] : []),
        { label: "Добавить карточку", onSelect: () => void handleAddCard(contextMenu) },
      ];
    }
    if (contextMenu.type === "relation") {
      return [
        { label: "Редактировать", onSelect: () => { openRelationModalForRelation(contextMenu.relationIds); closeContextMenu(); } },
        ...(relations.some((relation) => relation.type === "parenthood" && contextMenu.relationIds.includes(relation.id)) ? [{ label: "Центрировать связь", onSelect: () => void centerRelation(contextMenu.relationIds) }] : []),
        { label: "Удалить", onSelect: () => requestDeleteRelations(contextMenu.relationIds) },
        ...(canUndo ? [{ label: "Отменить действие", onSelect: () => void handleUndo() }] : []),
      ];
    }
    return [
      { label: "Редактировать", onSelect: () => handleEditCard(contextMenu.cardId) },
      { label: "Удалить", onSelect: () => handleDeleteCard(contextMenu.cardId) },
      ...(canUndo ? [{ label: "Отменить действие", onSelect: () => void handleUndo() }] : []),
    ];
  }

  function getContextMenuPosition(
    menu: ContextMenuState,
  ): ContextMenuPosition {
    return { x: menu.x, y: menu.y };
  }

  function renderContextMenu() {
    if (!contextMenu) return null;

    return (
      <ContextMenu
        position={getContextMenuPosition(contextMenu)}
        items={getContextMenuItems()}
        onClose={closeContextMenu}
      />
    );
  }

  function renderCanvasCards() {
    return cards.map((card) => (
      <CanvasCard
        key={card.id}
        card={card}
        viewport={viewport}
        locked={card.locked}
        showAuxiliaryLines={showAuxiliaryLines}
        onDeleteRequest={handleDeleteCard}
        onEditRequest={handleEditCard}
        onContextMenu={handleCardContextMenu}
        onSelectionToggle={handleSelectionToggle}
        onMoveFinished={async (cardId, movedAsGroup) => {
          if (movedAsGroup) await realignParentGroupsForCard(cardId);
        }}
      />
    ));
  }

  return (
    <div
      ref={canvasRef}
      className="canvas"
      onContextMenu={handleContextMenu}
      onClick={handleBackgroundClick}
      onPointerDown={handleCanvasPointerDown}
      onPointerMove={handleCanvasPointerMove}
      onPointerUp={handleCanvasPointerUp}
      onPointerCancel={handleCanvasPointerUp}
      onWheel={handleWheel}
    >
      <GenerationGuides cards={cards} panY={viewport.pan.y} zoom={viewport.zoom} visible={showAuxiliaryLines} />
      <div className="canvas-world" style={{ transform }}>
        {showAuxiliaryLines && lockedAreas.map((area) => {
          const bounds = area.bounds;
          if (!bounds) return null;
          const padding = 32;
          return <div key={area.id} className="canvas-locked-area" style={{ left: bounds.left - padding, top: bounds.top - padding, width: bounds.right - bounds.left + padding * 2, height: bounds.bottom - bounds.top + padding * 2 }} title="Заблокированная область" aria-label="Заблокированная область" />;
        })}
        {lockedAreaPreview && <div className="canvas-locked-area-preview" style={{ left: lockedAreaPreview.left, top: lockedAreaPreview.top, width: lockedAreaPreview.width, height: lockedAreaPreview.height }} aria-label="Просматриваемая заблокированная область" />}
        <ParenthoodOverlays cards={cards} relations={relations} selectedRelationIds={selectedRelationIds} onRelationSelect={selectRelation} onRelationDoubleClick={openRelationModalForRelation} onRelationContextMenu={openRelationMenu} />
        <MarriageOverlays cards={cards} relations={relations} selectedRelationIds={selectedRelationIds} onRelationSelect={(relationId) => selectRelation([relationId])} onRelationDoubleClick={(relationId) => openRelationModalForRelation([relationId])} onRelationContextMenu={(relationId, position) => openRelationMenu([relationId], position)} onCouplePhotoClick={openCouplePhotoModal} />
        {renderCanvasCards()}
      </div>
      <CanvasControls
        zoom={viewport.zoom}
        onZoomIn={zoomIn}
        onZoomOut={zoomOut}
        onZoomTo100={zoomTo100}
        onFitTree={handleFitTree}
      />
      {areaSelection && <div className="canvas-area-selection" style={{ left: areaSelection.left, top: areaSelection.top, width: areaSelection.width, height: areaSelection.height }}><span>{areaSelection.cardIds.length ? `Выбрана область: ${areaSelection.cardIds.length} карточек` : "Выбрана пустая область"}</span>{areaSelection.worldBounds && <button type="button" onClick={(event) => { event.stopPropagation(); const { cardIds, worldBounds } = areaSelection; if (!worldBounds) return; void setCardsLocked(cardIds, true).then(() => { saveLockedArea(cardIds, worldBounds); clearAreaSelection(); }); }}>Зафиксировать область</button>}</div>}
      {selectionActionPosition && <button type="button" className="canvas-selection-action" style={{ left: selectionActionPosition.x + 10, top: selectionActionPosition.y + 10 }} onClick={(event) => { event.stopPropagation(); openRelationModalFromSelection(); }}>Создать связь</button>}
      {renderContextMenu()}
      {personModal && (() => { const card = personModal.id ? cards.find((item) => item.id === personModal.id) : undefined; if (personModal.extended && card) return <PersonExtendedModal card={card} cards={cards} relations={relations} onSave={async (person) => { await savePerson(person); }} onDelete={() => handleDeleteCard(card.id)} onToggleLock={() => void togglePersonLock(card.id, Boolean(card.locked))} onClose={closePersonModal} />; return <PersonShortModal personId={card?.id ?? ""} person={card?.person ?? {}} sourceKey={personModal.isNew ? "new" : personModal.id ?? "person"} isNew={personModal.isNew} locked={Boolean(card?.locked)} onSave={savePerson} onDelete={() => { if (card) handleDeleteCard(card.id); }} onExtendedEdit={() => setPersonModal((current) => current ? { ...current, extended: true } : current)} onToggleLock={card ? () => void togglePersonLock(card.id, Boolean(card.locked)) : undefined} onClose={closePersonModal} />; })()}
      {relationModal && (() => { const relation = relationModal.relationIds[0] ? relations.find((item) => item.id === relationModal.relationIds[0]) : undefined; const relationCards = relationModal.personIds.map((id) => cards.find((card) => card.id === id)).filter((card): card is typeof cards[number] => Boolean(card)); if (relationModal.extended && relation) return <RelationExtendedModal relation={relation} cards={relationCards} parentIds={relationModal.parentIds} onSave={saveRelationDetails} onDelete={() => requestDeleteRelations(relationModal.relationIds)} onClose={() => { setRelationModal(null); selectCard(null); }} />; return <RelationShortModal type={relationModal.type} cards={relationCards} parentIds={relationModal.parentIds} isNew={relationModal.isNew} status={relationModal.status} autoAlign={relationModal.autoAlign} onAutoAlignChange={setParenthoodAutoAlign} onSave={saveRelation} onDelete={() => requestDeleteRelations(relationModal.relationIds)} onExtendedEdit={() => setRelationModal((current) => current ? { ...current, extended: true } : current)} onClose={() => { setRelationModal(null); selectCard(null); }} />; })()}
      {couplePhotoModal && <CouplePhotoModal relationId={couplePhotoModal.relationId} personId={couplePhotoModal.personId} currentUrl={couplePhotoModal.currentUrl} onSave={saveCouplePhoto} onClose={() => setCouplePhotoModal(null)} />}
      {confirmation && createPortal(<ConfirmActionModal title={confirmation.title} message={confirmation.message} confirmLabel={confirmation.confirmLabel} onConfirm={() => { const action = confirmation.action; setConfirmation(null); action(); }} onClose={() => setConfirmation(null)} />, document.body)}
    </div>
  );
}
