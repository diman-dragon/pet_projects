import type { CreateCardInput } from "../../services/db/CardService";

export const EMPTY_CARD: CreateCardInput = {
  front: "",
  back: "",
  context: "",
  transcription: "",
  etymology: "",
  mnemonic: "",
  notes: "",
  example_1: "",
  example_2: "",
  example_3: "",
  audio_path: "",
  image_path: "",
  video_path: "",
  source: "user",
  tags: ""
};

export function deckPalette(index: number) {
  const palette = ["#2563eb", "#0f766e", "#7c3aed", "#c2410c", "#be123c", "#0891b2"];
  return palette[index % palette.length];
}
