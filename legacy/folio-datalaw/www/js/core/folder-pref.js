// Chosen library folder ({uri, name} or null), persisted.
import { store } from './storage.js';
import { emit } from './bus.js';
let cur = null;
try { cur = JSON.parse(store.get('folder') ?? 'null'); } catch { cur = null; }
export const getFolder = () => cur;
export function setFolder(f) { cur = f; store.set('folder', JSON.stringify(f)); emit('folderchange', f); }
