import type { GrammarExercise, GrammarExerciseType, GrammarUnit } from "../../../types/domain";
import type { GrammarUnitInput } from "../../../services/db/GrammarService";

export const emptyUnit: GrammarUnitInput = {
  title: "",
  language: "English",
  level: "A1",
  section: "",
  tags: "",
  theory: "",
  examples: "",
};

export const demoExercises: Array<{
  type: GrammarExerciseType;
  prompt: string;
  payload: string;
  answer: string;
  explanation: string;
}> = [
  {
    type: "choice",
    prompt: "Choose the correct article: ___ apple a day keeps the doctor away.",
    payload: JSON.stringify({ options: ["A", "An", "The"] }),
    answer: "An",
    explanation: "Use 'an' before a vowel sound: an apple.",
  },
  {
    type: "fill_blank",
    prompt: "Fill the blank: I saw ___ old friend yesterday.",
    payload: "{}",
    answer: "an",
    explanation: "'Old' starts with a vowel sound, so the article is 'an'.",
  },
  {
    type: "reorder",
    prompt: "Build the phrase.",
    payload: JSON.stringify({ blocks: ["an", "interesting", "book"] }),
    answer: "an interesting book",
    explanation: "The article comes before the adjective and noun.",
  },
  {
    type: "translate",
    prompt: "Translate: Это интересная книга.",
    payload: "{}",
    answer: "It is an interesting book.",
    explanation: "Use 'an' before 'interesting' because it begins with a vowel sound.",
  },
];

export function normalize(value: string) {
  return value.trim().toLowerCase().replace(/[.!?]+$/g, "");
}

export function parsePayload(exercise: GrammarExercise) {
  try {
    return JSON.parse(exercise.payload || "{}") as { options?: string[]; blocks?: string[] };
  } catch {
    return {};
  }
}

export function statusLabel(status: GrammarUnit["status"]) {
  if (status === "mastered") return "Освоено";
  if (status === "in_progress") return "В процессе";
  return "Не начато";
}
