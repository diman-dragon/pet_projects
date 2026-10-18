// Language state (auto | ru | en), t() lookup and DOM translation via data-i18n attributes.
import { store } from './storage.js';
import { emit } from './bus.js';
import { ru } from './locales/ru.js';
import { en } from './locales/en.js';
const dict = { ru, en };
let pref = store.get('lang') ?? 'auto';
export const getLang = () => pref;
export const resolve = () => (pref !== 'auto' ? pref : (navigator.language || 'en').toLowerCase().startsWith('ru') ? 'ru' : 'en');
export const t = (k) => dict[resolve()][k] ?? k;
export function applyLang() {
  document.documentElement.lang = resolve();
  document.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll('[data-i18n-aria]').forEach((el) => el.setAttribute('aria-label', t(el.dataset.i18nAria)));
  emit('langchange', resolve());
}
export function setLang(v) { pref = v; store.set('lang', v); applyLang(); }
