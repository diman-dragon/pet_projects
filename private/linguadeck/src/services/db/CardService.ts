import type { Card, CardDraft } from "../../types/domain";
import { invokeDb } from "./invokeDb";

export type CreateCardInput = CardDraft;
export type { Card } from "../../types/domain";

function encodeTags(tags?: string[] | string | null) {
  if (!tags) return null;
  return Array.isArray(tags) ? tags.join(", ") : tags;
}

function toPayload(input: Partial<CreateCardInput>) {
  return { ...input, tags: encodeTags(input.tags) };
}

export const CardService = {
  createExtended(deckId: number, input: CreateCardInput) {
    return invokeDb("card_create", { deckId, input: toPayload(input) });
  },

  /** Single Rust transaction — drastically faster than looping createExtended */
  createBatch(deckId: number, inputs: CreateCardInput[]): Promise<number> {
    return invokeDb<number>("card_create_batch", {
      deckId,
      inputs: inputs.map(toPayload),
    });
  },

  update(cardId: number, input: CreateCardInput) {
    return invokeDb("card_update", { cardId, input: toPayload(input) });
  },

  getByDeckId(deckId: number) {
    return invokeDb<Card[]>("card_list", { deckId });
  },

  delete(cardId: number, deckId: number) {
    return invokeDb("card_delete", { cardId, deckId });
  },

  deleteMany(cardIds: number[], deckId: number) {
    return invokeDb("card_delete_many", { cardIds, deckId });
  },
};
