// Viewer toolbar: zoom buttons, reading-mode toggle and the page indicator (tap = go to page) bound to the active handle.
import { t } from '../core/i18n.js';
import { getReadMode, setReadMode } from '../core/readmode.js';
import { on } from '../core/bus.js';
import { icon } from './icon.js';
import { toast } from './toast.js';
const pageBtn = document.getElementById('viewer-page');
const modeBtn = document.getElementById('mode-btn');
let handle = null;
let total = 0;
export const setHandle = (h) => { handle = h; if (!h) pageBtn.textContent = ''; };
export function showPage(n, count) { total = count; pageBtn.textContent = `${n} / ${count}`; }
const paintMode = () => { modeBtn.innerHTML = icon(getReadMode() === 'paged' ? 'pages' : 'scroll'); };
document.getElementById('zoom-in').addEventListener('click', () => handle?.zoomBy(1.25));
document.getElementById('zoom-out').addEventListener('click', () => handle?.zoomBy(0.8));
modeBtn.addEventListener('click', () => {
  setReadMode(getReadMode() === 'paged' ? 'scroll' : 'paged');
  toast(t(getReadMode() === 'paged' ? 'set.modePaged' : 'set.modeScroll'));
});
on('modechange', (m) => { paintMode(); handle?.setMode?.(m); });
pageBtn.addEventListener('click', () => {
  const n = parseInt(prompt(`${t('viewer.goto')} (1-${total})`) ?? '', 10);
  if (n) handle?.goTo(n);
});
paintMode();
