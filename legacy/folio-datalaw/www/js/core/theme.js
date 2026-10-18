// Theme state: light | dark. Persists and applies to <html data-theme>.
import { store } from './storage.js';
import { emit } from './bus.js';
let cur = store.get('theme') ?? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
export const getTheme = () => cur;
export const applyTheme = () => { document.documentElement.dataset.theme = cur; emit('themechange', cur); };
export function setTheme(v) { cur = v; store.set('theme', v); applyTheme(); }
