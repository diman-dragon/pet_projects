import type { ViewType } from "../../types/domain";
import { invokeDb } from "./invokeDb";

export const ViewService = {
  /**
   * Log a card view event with optional time tracking.
   * timeSpentMs — сколько мс пользователь смотрел на текущую сторону карточки.
   * Fire-and-forget: никогда не бросает исключений.
   */
  logView(cardId: number, deckId: number, viewType: ViewType, timeSpentMs?: number) {
    const payload = {
      cardId,
      deckId,
      viewType,
      timeSpentMs: timeSpentMs ?? null,
    };
    console.log("[ViewService.logView] payload:", JSON.stringify(payload));
    return invokeDb("view_log", payload).catch((err: unknown) => {
      console.warn("[ViewService] logView failed", err);
    });
  },
};
