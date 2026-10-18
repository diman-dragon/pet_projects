// "Keep screen on while reading" setting: off | 5 | 15 (minutes without touches) | always.
import { store } from './storage.js';
import { emit } from './bus.js';
const OK = ['off', '5', '15', 'always'];
let cur = OK.includes(store.get('awake')) ? store.get('awake') : '5';
export const getAwake = () => cur;
export const awakeMinutes = () => ({ off: 0, 5: 5, 15: 15, always: Infinity })[cur];
export function setAwake(v) { cur = OK.includes(v) ? v : '5'; store.set('awake', cur); emit('awakechange', cur); }
