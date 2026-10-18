import { useState } from "react";
import type { ContextMenuPosition, ContextMenuState } from "@/types/contextMenu";

export function useCanvasContextMenus() {
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);

  return {
    contextMenu,
    openCanvasMenu: (position: ContextMenuPosition) => setContextMenu({ type: "canvas", ...position }),
    openCardMenu: (cardId: string, position: ContextMenuPosition) => setContextMenu({ type: "card", cardId, ...position }),
    openRelationMenu: (relationIds: string[], position: ContextMenuPosition) => setContextMenu({ type: "relation", relationIds, ...position }),
    closeContextMenu: () => setContextMenu(null),
  };
}
