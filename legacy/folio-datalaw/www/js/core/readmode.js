// Reading mode setting (scroll | paged): continuous ribbon or one page per screen with left/right swipes.
import { store } from './storage.js';
import { emit } from './bus.js';
let cur = store.get('readmode') === 'paged' ? 'paged' : 'scroll';
export const getReadMode = () => cur;
export function setReadMode(v) { cur = v === 'paged' ? 'paged' : 'scroll'; store.set('readmode', cur); emit('modechange', cur); }
