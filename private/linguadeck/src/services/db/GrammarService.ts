import type {
  Card,
  GrammarDashboard,
  GrammarExercise,
  GrammarExerciseType,
  GrammarSessionResult,
  GrammarUnit,
} from "../../types/domain";
import { invokeDb } from "./invokeDb";

export interface GrammarUnitInput {
  title: string;
  language: string;
  level: string;
  section: string;
  tags?: string | null;
  theory: string;
  examples: string;
}

export interface GrammarExerciseInput {
  unitId: number;
  type: GrammarExerciseType;
  prompt: string;
  payload: string;
  answer: string;
  explanation: string;
}

export const GrammarService = {
  dashboard() {
    return invokeDb<GrammarDashboard>("grammar_dashboard");
  },

  listUnits() {
    return invokeDb<GrammarUnit[]>("grammar_unit_list");
  },

  createUnit(input: GrammarUnitInput) {
    return invokeDb<{ id: number }>("grammar_unit_create", { input });
  },

  updateUnit(unitId: number, input: GrammarUnitInput) {
    return invokeDb("grammar_unit_update", { unitId, input });
  },

  deleteUnit(unitId: number) {
    return invokeDb("grammar_unit_delete", { unitId });
  },

  listExercises(unitId: number) {
    return invokeDb<GrammarExercise[]>("grammar_exercise_list", { unitId });
  },

  saveExercise(input: GrammarExerciseInput) {
    return invokeDb<{ id: number }>("grammar_exercise_save", { input });
  },

  practice(unitId: number, limit = 8) {
    return invokeDb<GrammarExercise[]>("grammar_practice", { unitId, limit });
  },

  completeSession(unitId: number, correctCount: number, totalCount: number) {
    return invokeDb<GrammarSessionResult>("grammar_session_complete", {
      unitId,
      correctCount,
      totalCount,
    });
  },

  linkedCards(unitId: number) {
    return invokeDb<Card[]>("grammar_linked_cards", { unitId });
  },

  unitsForCard(cardId: number) {
    return invokeDb<GrammarUnit[]>("grammar_units_for_card", { cardId });
  },

  async importModules(modules: import("../../data/englishGrammar").GrammarModule[]) {
    for (const module of modules) {
      const exists = await invokeDb<{ count: number }>("grammar_check_unit", { title: module.title });
      if (exists.count > 0) continue;
      const created = await this.createUnit({
        title: module.title,
        language: module.language,
        level: module.level,
        section: module.section,
        tags: module.tags,
        theory: module.theory,
        examples: module.examples,
      });
      await Promise.all(module.exercises.map((ex) => this.saveExercise({ ...ex, unitId: created.id })));
    }
  },
};
