import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { getNonOverlappingCardX } from "@/lib/canvas/getNonOverlappingCardX";
import { getGenerationCenterY } from "@/lib/canvas/getGenerationCenterY";
import { GENERATION_HEIGHT } from "@/lib/canvas/generationConstants";
import { getCenteredCardTargets } from "@/lib/canvas/centerCardLayout";
import { getMovementMemberIds } from "@/lib/canvas/getMovementMemberIds";
import { getSafeGroupDelta } from "@/lib/canvas/getSafeGroupDelta";
import { loadCanvasCardRecords, loadRelationRecords, mapCanvasCardRecord, mapRelationRecord, persistCanvasCardPosition, persistCanvasCardPositions, persistCanvasCardsLock } from "@/store/canvasPersistence";
import { loadStoredLockedAreas } from "@/lib/canvas/lockedAreaStorage";
import { PERSON_CARD_HEIGHT, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import type { CanvasCardData } from "@/types/canvasCard";
import type { RelationData } from "@/types/marriage";
import type { PersonData } from "@/types/person";

type CanvasCardsState = {
  cards: CanvasCardData[];
  selectedCardId: string | null;
  selectedCardIds: string[];
  relations: RelationData[];
  loadRelations: () => Promise<void>;
  loadGraph: () => Promise<void>;
  loadCards: () => Promise<void>;
  addCard: (x: number, y: number) => Promise<string>;
  moveCard: (id: string, x: number, y: number, moveAsGroup?: boolean) => void;
  beginMoveCard: (id: string, moveAsGroup?: boolean) => void;
  cancelMoveCard: (id: string) => void;
  finishMoveCard: (id: string, x: number, y: number, moveAsGroup?: boolean) => Promise<void>;
  alignCardsToCenter: (ids: string[], targetCenterX: number, targetCenterY?: number, anchorIds?: string[]) => Promise<void>;
  removeCard: (id: string) => Promise<void>;
  createRelation: (input: RelationCommandInput) => Promise<string>;
  updateRelation: (input: RelationCommandInput) => Promise<void>;
  updateParenthoodAutoAlign: (relationIds: string[], autoAlign: boolean) => Promise<void>;
  deleteRelation: (id: string) => Promise<void>;
  updatePerson: (id: string, person: PersonData) => Promise<void>;
  setCardsLocked: (ids: string[], locked: boolean) => Promise<void>;
  unlockAllCards: () => Promise<void>;
  selectCard: (id: string | null) => void;
  toggleCardSelection: (id: string) => void;
  canUndo: boolean;
  undoCount: number;
  undoHistory: string[];
  undoRevision: number;
  undoLastAction: () => Promise<void>;
  undoActions: (count: number) => Promise<void>;
};

type RelationCommandInput = { relationId?: string; relationType: "marriage" | "parenthood"; personAId: string; personBId: string; detailsJson?: string; allowExistingMarriage?: boolean };
type UndoAction = { kind: "delete" | "add"; card: CanvasCardData } | { kind: "move"; cards: CanvasCardData[] } | { kind: "lock"; ids: string[]; locked: boolean } | { kind: "relation-create"; relationId: string } | { kind: "relation-update" | "relation-delete"; relation: RelationData } | { kind: "parenthood-auto-align"; states: Array<{ id: string; autoAlign: boolean }> } | { kind: "person-update"; id: string; person: PersonData };

const undoStack: UndoAction[] = [];
const undoLabels: string[] = [];
const moveSnapshots = new Map<string, CanvasCardData[]>();
function pushUndo(action: UndoAction, label: string) { undoStack.push(action); undoLabels.push(label); }

function intersectsLockedArea(cards: CanvasCardData[], memberIds: string[], dx: number, dy: number) {
  const lockedAreas = loadStoredLockedAreas();
  return lockedAreas.some((area) => memberIds.some((id) => {
    const card = cards.find((item) => item.id === id);
    if (!card) return false;
    const left = card.x + dx;
    const right = left + PERSON_CARD_WIDTH;
    const top = card.y + dy - PERSON_CARD_HEIGHT / 2;
    const bottom = top + PERSON_CARD_HEIGHT;
    return left < area.bounds.right && right > area.bounds.left && top < area.bounds.bottom && bottom > area.bounds.top;
  }));
}

export const useCanvasCardsStore = create<CanvasCardsState>((set) => ({
  cards: [],
  selectedCardId: null,
  selectedCardIds: [],
  relations: [],
  loadRelations: async () => {
    const records = await loadRelationRecords();
    set({ relations: records.map(mapRelationRecord) });
  },
  loadGraph: async () => {
    await useCanvasCardsStore.getState().loadCards();
    await useCanvasCardsStore.getState().loadRelations();
  },
  canUndo: false,
  undoCount: 0,
  undoHistory: [],
  undoRevision: 0,
  loadCards: async () => {
    const records = await loadCanvasCardRecords();
    const normalizedRecords = records.map((record) => ({
      record,
      generationY: getGenerationCenterY(record.y),
    }));
    set({
      cards: normalizedRecords.map(({ record }) => mapCanvasCardRecord(record)),
    });
    const repairs = normalizedRecords.filter(({ record, generationY }) => Math.abs(record.y - generationY) > 0.01);
    if (repairs.length) {
      await Promise.all(repairs.map(({ record, generationY }) => persistCanvasCardPosition({ id: record.id, x: record.x, y: generationY })));
    }
  },
  addCard: async (x, y) => {
    const generationY = getGenerationCenterY(y);
    const id = await invoke<string>("create_canvas_card", {
      input: { x, y: generationY, generation: Math.round(generationY / GENERATION_HEIGHT) },
    });
    const safeX = getNonOverlappingCardX(useCanvasCardsStore.getState().cards, id, x, generationY);
    const newCard = { id, x: safeX, y: generationY, person: {} };
    pushUndo({ kind: "add", card: newCard }, "Добавление карточки");
    set((state) => {
      const generationY = getGenerationCenterY(y);
      const card = { id, x, y: generationY, person: {} };
      return {
        cards: [
          ...state.cards,
          {
            ...card,
            x: safeX,
          },
        ],
        canUndo: true,
        undoCount: undoStack.length,
        undoHistory: [...undoLabels],
      };
    });
    if (safeX !== x) {
      void persistCanvasCardPosition({ id, x: safeX, y: generationY });
    }
    return id;
  },
  moveCard: (id, x, y, moveAsGroup = true) =>
    set((state) => {
      const memberIds = getMovementMemberIds(state, id, moveAsGroup);
      const activeCard = state.cards.find((card) => card.id === id);
      if (!activeCard || activeCard.locked) return state;
      const movableIds = memberIds.filter((memberId) => !state.cards.find((card) => card.id === memberId)?.locked);
      const groupDelta = getSafeGroupDelta(state.cards, movableIds, x, y, id);
      const dx = groupDelta.x;
      const dy = groupDelta.y;
      return { cards: state.cards.map((card) => movableIds.includes(card.id) ? { ...card, x: card.x + dx, y: getGenerationCenterY(card.y + dy) } : card) };
    }),
  beginMoveCard: (id, moveAsGroup = true) => {
    const state = useCanvasCardsStore.getState();
    const memberIds = getMovementMemberIds(state, id, moveAsGroup);
    if (state.cards.find((card) => card.id === id)?.locked) return;
    moveSnapshots.set(id, state.cards.filter((card) => memberIds.includes(card.id) && !card.locked).map((card) => ({ ...card, person: { ...card.person } })));
  },
  cancelMoveCard: (id) => {
    moveSnapshots.delete(id);
  },
  finishMoveCard: async (id, x, y, moveAsGroup = true) => {
    const currentState = useCanvasCardsStore.getState();
    const snapshot = moveSnapshots.get(id);
    if (currentState.cards.find((card) => card.id === id)?.locked) { moveSnapshots.delete(id); return; }
    const memberIds = getMovementMemberIds(currentState, id, moveAsGroup).filter((memberId) => !currentState.cards.find((card) => card.id === memberId)?.locked);
    const generationY = getGenerationCenterY(y);
    // При Alt сохраняем именно выбранную пользователем координату. Соседние
    // карточки проверяются ниже, но не заменяем дальнее перемещение ближайшим
    // свободным кандидатом (это возвращало супруга к минимальному зазору).
    // The minimum card gap is invariant for ordinary and Alt movement alike.
    // Alt changes grouping only; it never permits a card to be left on another.
    const safeX = getNonOverlappingCardX(useCanvasCardsStore.getState().cards, id, x, generationY, memberIds);
    const groupDelta = getSafeGroupDelta(currentState.cards, memberIds, safeX, generationY, id);
    const activeDx = groupDelta.x;
    const activeDy = groupDelta.y;
    if (intersectsLockedArea(currentState.cards, memberIds, activeDx, activeDy)) {
      if (snapshot) set((state) => ({ cards: state.cards.map((card) => snapshot.find((previous) => previous.id === card.id) ?? card) }));
      moveSnapshots.delete(id);
      return;
    }
    const proposedCards = currentState.cards.map((card) => memberIds.includes(card.id) ? { ...card, x: card.x + activeDx, y: getGenerationCenterY(card.y + activeDy) } : card);
    const finalCards = proposedCards.filter((card) => memberIds.includes(card.id));
    await persistCanvasCardPositions(finalCards);
    const finalById = new Map(finalCards.map((card) => [card.id, card]));
    set((state) => {
      return {
        cards: state.cards.map((card) =>
          finalById.has(card.id) ? { ...card, x: finalById.get(card.id)!.x, y: finalById.get(card.id)!.y } : card,
        ),
        canUndo: undoStack.length > 0,
      };
    });
    const changed = Boolean(snapshot?.some((previous) => finalById.get(previous.id)?.x !== previous.x || finalById.get(previous.id)?.y !== previous.y));
    if (changed && snapshot) {
      pushUndo({ kind: "move", cards: snapshot }, "Перемещение карточки");
      set({ undoCount: undoStack.length, undoHistory: [...undoLabels] });
    }
    moveSnapshots.delete(id);
  },
  alignCardsToCenter: async (ids, targetCenterX, targetCenterY, anchorIds = ids) => {
    const state = useCanvasCardsStore.getState();
    const targets = getCenteredCardTargets(state.cards, state.relations, ids, targetCenterX, targetCenterY, anchorIds);
    if (!targets) return;
    const selected = state.cards.filter((card) => ids.includes(card.id) && !card.locked);
    if (!selected.length) return;
    const hasMovement = selected.some((card) => {
      const target = targets.get(card.id);
      return target && (Math.abs(target.x - card.x) > 0.01 || Math.abs(target.y - card.y) > 0.01);
    });
    if (!hasMovement) return;
    const snapshot = selected.map((card) => ({ ...card, person: { ...card.person } }));
    await persistCanvasCardPositions(selected.map((card) => { const target = targets.get(card.id) ?? { x: card.x, y: card.y }; return { id: card.id, x: target.x, y: target.y }; }));
    set((current) => ({ cards: current.cards.map((card) => {
      const target = targets.get(card.id);
      return target ? { ...card, x: target.x, y: target.y } : card;
    }) }));
    pushUndo({ kind: "move", cards: snapshot }, "Выравнивание группы детей");
    set({ canUndo: true, undoCount: undoStack.length, undoHistory: [...undoLabels] });
  },
  removeCard: async (id) => {
    const card = useCanvasCardsStore.getState().cards.find((item) => item.id === id);
    await invoke("delete_canvas_card", { id });
    if (card) pushUndo({ kind: "delete", card }, "Удаление карточки");
    set((state) => ({
      cards: state.cards.filter((card) => card.id !== id),
      relations: state.relations.filter((relation) => relation.personAId !== id && relation.personBId !== id),
      canUndo: undoStack.length > 0,
      undoCount: undoStack.length,
      undoHistory: [...undoLabels],
      selectedCardId:
        state.selectedCardId === id ? null : state.selectedCardId,
      selectedCardIds: state.selectedCardIds.filter((cardId) => cardId !== id),
    }));
  },
  createRelation: async (input) => {
    const relationId = await invoke<string>("create_relation", input);
    pushUndo({ kind: "relation-create", relationId }, "Создание связи");
    await useCanvasCardsStore.getState().loadGraph();
    set({ canUndo: true, undoCount: undoStack.length, undoHistory: [...undoLabels] });
    return relationId;
  },
  updateRelation: async (input) => {
    const current = useCanvasCardsStore.getState().relations.find((relation) => input.relationId ? relation.id === input.relationId : relation.personAId === input.personAId && relation.personBId === input.personBId && relation.type === input.relationType);
    if (!current) return;
    await invoke("update_relation", { relationId: current.id, relationType: input.relationType, personAId: input.personAId, personBId: input.personBId, detailsJson: input.detailsJson, allowExistingMarriage: input.allowExistingMarriage });
    pushUndo({ kind: "relation-update", relation: current }, "Изменение связи");
    await useCanvasCardsStore.getState().loadGraph();
    set({ canUndo: true, undoCount: undoStack.length, undoHistory: [...undoLabels] });
  },
  updateParenthoodAutoAlign: async (relationIds, autoAlign) => {
    const uniqueIds = [...new Set(relationIds)];
    const current = useCanvasCardsStore.getState().relations.filter((relation) => uniqueIds.includes(relation.id) && relation.type === "parenthood");
    if (!current.length) return;
    await invoke("update_parenthood_auto_align_family", { relationIds: current.map((relation) => relation.id), autoAlign });
    pushUndo({ kind: "parenthood-auto-align", states: current.map((relation) => ({ id: relation.id, autoAlign: relation.details?.autoAlign !== false })) }, autoAlign ? "Включение автоцентрирования детей" : "Отключение автоцентрирования детей");
    await useCanvasCardsStore.getState().loadGraph();
    set({ canUndo: true, undoCount: undoStack.length, undoHistory: [...undoLabels] });
  },
  deleteRelation: async (id) => {
    const relation = useCanvasCardsStore.getState().relations.find((item) => item.id === id);
    await invoke("delete_relation", { relationId: id });
    if (relation) pushUndo({ kind: "relation-delete", relation }, "Удаление связи");
    await useCanvasCardsStore.getState().loadGraph();
    set({ canUndo: true, undoCount: undoStack.length, undoHistory: [...undoLabels] });
  },
  updatePerson: async (id, person) => {
    const current = useCanvasCardsStore.getState().cards.find((card) => card.id === id);
    if (!current) return;
    await invoke("update_person", { input: { id, sex: person.sex, first_name: person.firstName, patronymic: person.patronymic, last_name: person.lastName, previous_last_name: person.previousLastName, birth_date: person.birthDate, death_date: person.deathDate, photo_url: person.photoUrl, photo_zoom: person.photoZoom, photo_offset_x: person.photoOffsetX, photo_offset_y: person.photoOffsetY, gender_identity: person.genderIdentity, nationality: person.nationality, ethnicity: person.ethnicity, religion: person.religion, occupation: person.occupation, education: person.education, description: person.description, biography: person.biography, research_status: person.researchStatus } });
    pushUndo({ kind: "person-update", id, person: { ...current.person } }, "Изменение персоны");
    await useCanvasCardsStore.getState().loadGraph();
    set({ canUndo: true, undoCount: undoStack.length, undoHistory: [...undoLabels] });
  },
  setCardsLocked: async (ids, locked) => {
    const uniqueIds = [...new Set(ids)];
    if (!uniqueIds.length) return;
    await persistCanvasCardsLock(uniqueIds, locked);
    pushUndo({ kind: "lock", ids: uniqueIds, locked }, locked ? "Фиксация области" : "Снятие фиксации области");
    set({ canUndo: true, undoCount: undoStack.length, undoHistory: [...undoLabels] });
    await useCanvasCardsStore.getState().loadCards();
  },
  unlockAllCards: async () => {
    const lockedIds = useCanvasCardsStore.getState().cards.filter((card) => card.locked).map((card) => card.id);
    await persistCanvasCardsLock([], false);
    if (lockedIds.length) pushUndo({ kind: "lock", ids: lockedIds, locked: false }, "Снятие фиксации со всех карточек");
    await useCanvasCardsStore.getState().loadCards();
    set({ canUndo: undoStack.length > 0, undoCount: undoStack.length, undoHistory: [...undoLabels] });
  },
  selectCard: (id) => set({ selectedCardId: id, selectedCardIds: id ? [id] : [] }),
  toggleCardSelection: (id) => set((state) => {
    const selectedCardIds = state.selectedCardIds.includes(id)
      ? state.selectedCardIds.filter((cardId) => cardId !== id)
      : [...state.selectedCardIds, id];
    return { selectedCardIds, selectedCardId: selectedCardIds[selectedCardIds.length - 1] ?? null };
  }),
  undoLastAction: async () => {
    const action = undoStack.pop();
    if (!action) return;
    undoLabels.pop();
    set({ canUndo: undoStack.length > 0, undoCount: undoStack.length, undoHistory: [...undoLabels] });
    if (action.kind === "delete") {
      await invoke("restore_canvas_card", { id: action.card.id });
      set((state) => ({ cards: [...state.cards, action.card] }));
      await useCanvasCardsStore.getState().loadRelations();
    } else if (action.kind === "add") {
      await invoke("purge_canvas_card", { id: action.card.id });
      set((state) => ({ cards: state.cards.filter((card) => card.id !== action.card.id) }));
    } else if (action.kind === "move") {
      await invoke("update_canvas_cards_positions", { positions: action.cards.map((card) => ({ id: card.id, x: card.x, y: card.y, generation: Math.round(card.y / GENERATION_HEIGHT) })) });
      set((state) => ({ cards: state.cards.map((card) => action.cards.find((previous) => previous.id === card.id) ?? card) }));
    } else if (action.kind === "lock") {
      await invoke("set_canvas_cards_locked", { ids: action.ids, locked: !action.locked });
      await useCanvasCardsStore.getState().loadCards();
    } else if (action.kind === "relation-create") {
      await invoke("delete_relation", { relationId: action.relationId });
      await useCanvasCardsStore.getState().loadGraph();
    } else if (action.kind === "relation-update") {
      await invoke("update_relation", { relationId: action.relation.id, relationType: action.relation.type, personAId: action.relation.personAId, personBId: action.relation.personBId, detailsJson: JSON.stringify(action.relation.details ?? {}), allowExistingMarriage: true });
      await useCanvasCardsStore.getState().loadGraph();
    } else if (action.kind === "relation-delete") {
      await invoke<string>("create_relation", { relationType: action.relation.type, personAId: action.relation.personAId, personBId: action.relation.personBId, detailsJson: JSON.stringify(action.relation.details ?? {}), allowExistingMarriage: true });
      await useCanvasCardsStore.getState().loadGraph();
    } else if (action.kind === "parenthood-auto-align") {
      const groups = new Map<boolean, string[]>();
      for (const state of action.states) groups.set(state.autoAlign, [...(groups.get(state.autoAlign) ?? []), state.id]);
      for (const [autoAlign, relationIds] of groups) await invoke("update_parenthood_auto_align_family", { relationIds, autoAlign });
      await useCanvasCardsStore.getState().loadGraph();
    } else if (action.kind === "person-update") {
      await invoke("update_person", { input: { id: action.id, sex: action.person.sex, first_name: action.person.firstName, patronymic: action.person.patronymic, last_name: action.person.lastName, previous_last_name: action.person.previousLastName, birth_date: action.person.birthDate, death_date: action.person.deathDate, photo_url: action.person.photoUrl, photo_zoom: action.person.photoZoom, photo_offset_x: action.person.photoOffsetX, photo_offset_y: action.person.photoOffsetY, gender_identity: action.person.genderIdentity, nationality: action.person.nationality, ethnicity: action.person.ethnicity, religion: action.person.religion, occupation: action.person.occupation, education: action.person.education, description: action.person.description, biography: action.person.biography, research_status: action.person.researchStatus } });
      await useCanvasCardsStore.getState().loadGraph();
    }
  },
  undoActions: async (count) => {
    const amount = Math.max(0, Math.min(count, useCanvasCardsStore.getState().undoCount));
    for (let index = 0; index < amount; index += 1) {
      await useCanvasCardsStore.getState().undoLastAction();
    }
  },
}));
