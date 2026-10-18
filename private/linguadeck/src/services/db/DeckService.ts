import { invokeDb } from "./invokeDb";
import type { DashboardStats, Deck } from "../../types/domain";

export type { Deck } from "../../types/domain";

export const DeckService = {
  create(
    courseId: number,
    name: string,
    description = "",
    icon = "Layers3",
    color = "#2563eb",
    language = ""
  ) {
    return invokeDb("deck_create", { courseId, name, description, icon, color, language });
  },

  update(
    deckId: number,
    payload: Pick<Deck, "name" | "description" | "icon" | "color" | "language">
  ) {
    return invokeDb("deck_update", { deckId, ...payload });
  },

  getByCourseId() {
    return invokeDb<Deck[]>("deck_list");
  },

  getAll() {
    return this.getByCourseId();
  },

  // FIX: убран console.log — он попадал в продакшн-сборку и замусоривал консоль
  async getDashboardStats() {
    return invokeDb<DashboardStats>("deck_dashboard_stats");
  },

  delete(deckId: number) {
    return invokeDb("deck_delete", { deckId });
  },
};
