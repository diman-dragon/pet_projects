// Reading "night mode" setting (on | off): inverts document pages when the dark theme is active.
import { store } from './storage.js';
import { emit } from './bus.js';
let cur = store.get('night') ?? 'on';
export const getNight = () => cur;
export function setNight(v) { cur = v; store.set('night', v); emit('nightchange', v); }
