// Viewer overlay lifecycle: open a document with its format module, close, remember the last page.
import { find } from '../formats/registry.js';
import { updateDoc } from '../data/library.js';
import { pushBack } from './back.js';
import { setHandle, showPage } from './viewer-toolbar.js';
import { initThumbs, updateThumbsActive, destroyThumbs } from './viewer-thumbs.js';
import { suspendPanel, restorePanel } from './tools-panel.js';
import { startAwake, stopAwake } from '../core/awake.js';
import { emit } from '../core/bus.js';
import { setActiveHandle, setActiveDoc } from './viewer-handle.js';
import { loadAllOcr } from '../data/ocr-store.js';
import { clearAnnots } from './annot-store.js';
import { clearSel } from './text-select.js';
import { t } from '../core/i18n.js';
import { getReadMode } from '../core/readmode.js';
const root = document.getElementById('viewer');
const pages = document.getElementById('viewer-pages');
const scroller = document.getElementById('viewer-scroll');
const status = document.getElementById('viewer-status');
let handle = null, popBack = null, session = 0, saveTimer;
const setStatus = (text) => { status.hidden = !text; status.textContent = text ?? ''; };
let pending = null;
const saveLater = (id, n) => { pending = { id, n }; clearTimeout(saveTimer); saveTimer = setTimeout(flushSave, 600); };
async function flushSave() {
  clearTimeout(saveTimer);
  const p = pending;
  pending = null;
  if (p) { await updateDoc(p.id, { lastPage: p.n }); emit('libchange'); }
}
export function closeViewer() {
  session++;
  stopAwake();
  flushSave();
  try { handle?.destroy(); } catch (err) { console.error(err); }
  handle = null;
  setHandle(null);
  setActiveHandle(null);
  setActiveDoc(null);
  emit('viewerclose');
  clearSel();
  clearAnnots();
  destroyThumbs();
  suspendPanel();
  popBack?.();
  popBack = null;
  root.hidden = true;
  root.classList.remove('no-thumbs', 'no-draw', 'no-mode');
  pages.replaceChildren();
  setStatus(null);
}
export async function openViewer(meta, blob) {
  closeViewer();
  const mine = session;
  document.getElementById('viewer-title').textContent = meta.name;
  setStatus(t('viewer.loading'));
  root.hidden = false;
  popBack = pushBack(closeViewer);
  try {
    const mod = find(meta.name, blob.type);
    const onPage = (n, total) => { showPage(n, total); saveLater(meta.id, n); updateThumbsActive(n); emit('pagechange', { n, total }); };
    const h = await mod.mount({ container: pages, scroller, blob, startPage: meta.lastPage ?? 1, mode: getReadMode(), onPage });
    if (mine !== session) { h.destroy(); return; }
    handle = h;
    setHandle(h);
    setActiveHandle(h);
    setActiveDoc(meta);
    const caps = { thumbs: true, draw: true, mode: true, ...h.caps };
    root.classList.toggle('no-thumbs', !caps.thumbs);
    root.classList.toggle('no-draw', !caps.draw);
    root.classList.toggle('no-mode', !caps.mode);
    if (h.setOcr) loadAllOcr(meta.id).then((list) => list.forEach(([i, d]) => h.setOcr(i, d))).catch(() => {});
    initThumbs(h.doc, h, meta.lastPage ?? 1);
    if (caps.draw) restorePanel();
    startAwake();
    setStatus(null);
  } catch (err) {
    console.error(err);
    if (mine === session) { pages.replaceChildren(); setStatus(t('viewer.error')); }
  }
}
document.getElementById('viewer-back').addEventListener('click', closeViewer);
