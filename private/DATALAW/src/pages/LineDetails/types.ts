// src/pages/LineDetails/types.ts
import { BaseEntity, ID as GlobalID, Appeal as GlobalAppeal } from '../../types/entities';

export type ID = GlobalID;

// Локально расширяем или определяем Appeal, если нужно
export type Appeal = GlobalAppeal;

/**
 * ОПРЕДЕЛЯЕМ LineFull ЗДЕСЬ
 * Так как в глобальном entities.ts его нет, мы описываем его структуру тут.
 */
export interface LineFull extends BaseEntity {
  case_id: number;
  title: string;
  description: string | null;
  position_index: number;
  folder_name: string | null;
  summary?: string | null;
}
