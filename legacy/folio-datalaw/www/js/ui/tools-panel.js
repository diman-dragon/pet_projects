// Floating tool palette (transparent backing, over the document): tools, colours, page flip, lock position, hide.
import { setDrawTool, setDrawColor } from './viewer-draw.js';
import { initPanelDrag } from './panel-drag.js';
import { getActiveHandle } from './viewer-handle.js';
import { toast } from './toast.js';
import { t } from '../core/i18n.js';
import { on, emit } from '../core/bus.js';
import { getPref, setPref } from '../core/prefs.js';
import { stylusOnly, setStylusOnly } from './stylus.js';
import { toggleOcr } from './ocr-menu.js';
const panel = document.getElementById('tools-panel');
const editBtn = document.getElementById('edit-btn');
const tools = [...panel.querySelectorAll('[data-tool]')];
const dots = panel.querySelectorAll('.color-dot[data-color]');
const custom = document.getElementById('tp-custom');
let pinned = getPref('panelpinned', false), tool = null, page = 1, total = 1;
const place = initPanelDrag(panel, () => pinned);
function pick(next) {
  tool = next;
  tools.forEach((b) => b.classList.toggle('active', b.dataset.tool === tool));
  setDrawTool(tool);
  if (tool === 'text-select' || tool === 'underline') toast(t('edit.hintMark'));
}
export function showPanel(open, { save = true } = {}) {
  panel.hidden = !open;
  editBtn.classList.toggle('active', open);
  if (save) setPref('panelopen', open);
  if (open) place(); else if (tool) pick(null);
  emit('editbar', open);
}
export const restorePanel = () => showPanel(getPref('panelopen', false), { save: false });
export const suspendPanel = () => showPanel(false, { save: false });
editBtn.addEventListener('click', () => showPanel(panel.hidden));
document.getElementById('tp-hide').addEventListener('click', () => showPanel(false));
const pinBtn = document.getElementById('tp-pin');
const paintPin = () => { pinBtn.classList.toggle('on', pinned); panel.classList.toggle('pinned', pinned); };
pinBtn.addEventListener('click', () => { pinned = !pinned; setPref('panelpinned', pinned); paintPin(); });
tools.forEach((b) => b.addEventListener('click', () => pick(tool === b.dataset.tool ? null : b.dataset.tool)));
const customDot = custom.parentElement;
function setColor(c, from) {
  dots.forEach((x) => x.classList.toggle('active', x === from));
  customDot.classList.toggle('active', from === customDot);
  setDrawColor(c);
  setPref('color', c);
}
dots.forEach((d) => d.addEventListener('click', () => setColor(d.dataset.color, d)));
custom.addEventListener('input', () => { customDot.style.setProperty('--c', custom.value); setColor(custom.value, customDot); });
const saved = getPref('color', null);
const found = [...dots].find((d) => d.dataset.color === saved);
if (saved && found) setColor(saved, found); else if (saved) { custom.value = saved; customDot.style.setProperty('--c', saved); setColor(saved, customDot); }
document.getElementById('tp-ocr').addEventListener('click', () => toggleOcr());
const flip = (d) => getActiveHandle()?.goTo(Math.min(total, Math.max(1, page + d)), true);
document.getElementById('tp-prev').addEventListener('click', () => flip(-1));
document.getElementById('tp-next').addEventListener('click', () => flip(1));
on('pagechange', (p) => { page = p.n; total = p.total; });
const penBtn = document.getElementById('tp-stylus');
const paintPen = () => { penBtn.classList.toggle('on', stylusOnly()); penBtn.setAttribute('aria-pressed', stylusOnly()); };
penBtn.addEventListener('click', () => { setStylusOnly(!stylusOnly()); toast(t(stylusOnly() ? 'edit.stylusOn' : 'edit.stylusOff')); });
on('stylus', () => { paintPen(); if (stylusOnly() && !tool) pick('pen'); });
paintPin();
paintPen();
