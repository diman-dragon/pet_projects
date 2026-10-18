// System Back: top layer (e.g. viewer) handles it first; otherwise the fallback runs.
import { plugin } from '../core/platform.js';
const stack = [];
let fallback = () => {};
export function pushBack(fn) {
  stack.push(fn);
  return () => { const i = stack.indexOf(fn); if (i >= 0) stack.splice(i, 1); };
}
export const setBackFallback = (fn) => { fallback = fn; };
plugin('App')?.addListener('backButton', () => (stack.at(-1) ?? fallback)());
