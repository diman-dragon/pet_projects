export type ContextMenuPosition = {
  x: number;
  y: number;
};

export type ContextMenuState = ContextMenuPosition &
  ({ type: "canvas" } | { type: "card"; cardId: string } | { type: "relation"; relationIds: string[] });
