// JSON preferences on top of the prefixed localStorage wrapper (panel position, thumbnails on/off, ...).
import { store } from './storage.js';
export function getPref(key, def) {
  try { const v = store.get(key); return v === null ? def : JSON.parse(v); } catch { return def; }
}
export const setPref = (key, value) => store.set(key, JSON.stringify(value));
