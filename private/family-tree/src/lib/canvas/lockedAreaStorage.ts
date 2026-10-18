import type { CanvasCardData } from "@/types/canvasCard";

export type LockedAreaBounds = { left: number; top: number; right: number; bottom: number };
export type StoredLockedArea = { id: string; cardIds: string[]; bounds: LockedAreaBounds };
const STORAGE_KEY = "family-tree-locked-areas";

export function loadStoredLockedAreas(): StoredLockedArea[] {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as StoredLockedArea[];
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

export function saveLockedArea(cardIds: string[], bounds: LockedAreaBounds) {
  const areas = loadStoredLockedAreas().filter((area) => !area.cardIds.some((id) => cardIds.includes(id)));
  areas.push({ id: `locked-area-${Date.now()}`, cardIds, bounds });
  localStorage.setItem(STORAGE_KEY, JSON.stringify(areas));
}

export function removeLockedAreas(cardIds: string[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(loadStoredLockedAreas().filter((area) => !area.cardIds.some((id) => cardIds.includes(id)))));
}

export function removeLockedArea(id: string) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(loadStoredLockedAreas().filter((area) => area.id !== id)));
}

export function getStoredAreaForCards(cards: CanvasCardData[]) {
  const ids = new Set(cards.map((card) => card.id));
  return loadStoredLockedAreas().find((area) => area.cardIds.length === ids.size && area.cardIds.every((id) => ids.has(id)));
}

export function isPointInsideLockedArea(x: number, y: number) {
  return loadStoredLockedAreas().some((area) => x >= area.bounds.left && x <= area.bounds.right && y >= area.bounds.top && y <= area.bounds.bottom);
}
