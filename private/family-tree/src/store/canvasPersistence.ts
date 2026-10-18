import { invoke } from "@tauri-apps/api/core";
import { getGenerationCenterY } from "@/lib/canvas/getGenerationCenterY";
import { GENERATION_HEIGHT } from "@/lib/canvas/generationConstants";
import type { CanvasCardData } from "@/types/canvasCard";
import type { RelationData, RelationType } from "@/types/marriage";

export type CanvasCardRecord = {
  id: string;
  x: number;
  y: number;
  generation: number;
  first_name?: string;
  patronymic?: string;
  last_name?: string;
  previous_last_name?: string;
  birth_date?: string;
  death_date?: string;
  photo_url?: string;
  photo_zoom?: number;
  photo_offset_x?: number;
  photo_offset_y?: number;
  sex?: "male" | "female";
  gender_identity?: string;
  nationality?: string;
  ethnicity?: string;
  religion?: string;
  occupation?: string;
  education?: string;
  description?: string;
  biography?: string;
  research_status?: string;
  locked?: boolean;
};

type RelationRecord = {
  id: string;
  union_type: RelationType;
  person_a_id: string;
  person_b_id: string;
  details_json?: string;
};

export async function loadCanvasCardRecords() {
  return invoke<CanvasCardRecord[]>("load_canvas_cards");
}

export async function loadRelationRecords() {
  return invoke<RelationRecord[]>("load_relations");
}

export function mapCanvasCardRecord(record: CanvasCardRecord): CanvasCardData {
  return {
    id: record.id,
    x: record.x,
    y: getGenerationCenterY(record.y),
    locked: record.locked,
    person: {
      firstName: record.first_name,
      patronymic: record.patronymic,
      lastName: record.last_name,
      previousLastName: record.previous_last_name,
      birthDate: record.birth_date,
      deathDate: record.death_date,
      photoUrl: record.photo_url,
      photoZoom: record.photo_zoom,
      photoOffsetX: record.photo_offset_x,
      photoOffsetY: record.photo_offset_y,
      sex: record.sex,
      genderIdentity: record.gender_identity,
      nationality: record.nationality,
      ethnicity: record.ethnicity,
      religion: record.religion,
      occupation: record.occupation,
      education: record.education,
      description: record.description,
      biography: record.biography,
      researchStatus: record.research_status,
    },
  };
}

export function mapRelationRecord(record: RelationRecord): RelationData {
  return {
    id: record.id,
    type: record.union_type,
    personAId: record.person_a_id,
    personBId: record.person_b_id,
    details: record.details_json ? JSON.parse(record.details_json) : undefined,
  };
}

type CanvasPosition = { id: string; x: number; y: number };

export async function persistCanvasCardPosition(position: CanvasPosition) {
  await invoke("update_canvas_card_position", {
    ...position,
    generation: Math.round(position.y / GENERATION_HEIGHT),
  });
}

export async function persistCanvasCardPositions(positions: CanvasPosition[]) {
  await invoke("update_canvas_cards_positions", {
    positions: positions.map((position) => ({
      ...position,
      generation: Math.round(position.y / GENERATION_HEIGHT),
    })),
  });
}

export async function persistCanvasCardsLock(ids: string[], locked: boolean) {
  await invoke("set_canvas_cards_locked", { ids, locked });
}
