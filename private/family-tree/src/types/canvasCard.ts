import type { PersonData } from "@/types/person";

export type CanvasCardData = {
  id: string;
  x: number;
  y: number;
  locked?: boolean;
  person: PersonData;
};
