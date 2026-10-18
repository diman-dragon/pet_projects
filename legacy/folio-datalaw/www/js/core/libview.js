// Library presentation setting (grid | list): covers grid or compact list.
import { store } from './storage.js';
import { emit } from './bus.js';
let cur = store.get('libview') === 'list' ? 'list' : 'grid';
export const getLibView = () => cur;
export function setLibView(v) { cur = v === 'list' ? 'list' : 'grid'; store.set('libview', cur); emit('viewchange', cur); }
