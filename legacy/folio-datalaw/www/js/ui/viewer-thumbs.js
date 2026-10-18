// Page thumbnails sidebar (transparent overlay, never moves the document): show / hide, pin = shown in every document.
import { pushBack } from './back.js';
import { getPref, setPref } from '../core/prefs.js';
const panel = document.getElementById('viewer-thumbs');
const list = document.getElementById('thumbs-list');
const btn = document.getElementById('thumbs-btn');
const tab = document.getElementById('thumbs-tab');
const pinBtn = document.getElementById('thumbs-pin');
let pinned = getPref('thumbspin', false);
const items = [];
let handle = null, io = null, cur = 1, active = null, popBack = null;
function want(it, on) {
  if (!it) return;
  it.want = on;
  if (!on) {
    if (it.canvas) { it.canvas.width = 0; it.canvas = null; it.box.replaceChildren(it.ph); }
    return;
  }
  if (it.canvas || it.busy || !handle) return;
  it.busy = true;
  const canvas = document.createElement('canvas');
  handle.thumb(it.num, canvas, 72, cur)
    .then(() => {
      it.busy = false;
      if (!it.want) { canvas.width = 0; return; }
      it.canvas = canvas;
      it.box.replaceChildren(canvas);
    })
    .catch(() => { it.busy = false; if (it.want && ++it.tries < 4) setTimeout(() => want(it, true), 400 * it.tries); });
}
function reveal(it) {
  const top = it.el.offsetTop, bot = top + it.el.offsetHeight;
  if (top < list.scrollTop + 8) list.scrollTop = top - 8;
  else if (bot > list.scrollTop + list.clientHeight - 8) list.scrollTop = bot - list.clientHeight + 8;
}
export function updateThumbsActive(n) {
  cur = n;
  active?.el.classList.remove('active');
  active = items[n - 1] ?? null;
  active?.el.classList.add('active');
  if (active && !panel.hidden) reveal(active);
}
export function toggleThumbs(force) {
  const show = force ?? panel.hidden;
  panel.hidden = !show;
  tab.hidden = show || !items.length;
  btn.classList.toggle('active', show);
  popBack?.();
  popBack = show ? pushBack(() => toggleThumbs(false)) : null;
  if (!show) return;
  updateThumbsActive(cur);
  for (let i = Math.max(0, cur - 3); i < Math.min(items.length, cur + 8); i++) want(items[i], true);
}
export function initThumbs(doc, h, start = 1) {
  destroyThumbs();
  if (!doc) return;
  handle = h;
  doc.getPage(1).then((p) => { const v = p.getViewport({ scale: 1 }); list.style.setProperty('--th', `${Math.round((72 * v.height) / v.width)}px`); });
  io = new IntersectionObserver((es) => {
    if (!panel.hidden) es.forEach((x) => want(items[x.target.dataset.idx], x.isIntersecting));
  }, { root: list, rootMargin: '300px' });
  for (let i = 1; i <= doc.numPages; i++) {
    const el = document.createElement('div');
    el.className = 'thumb-item';
    el.dataset.idx = i - 1;
    const box = Object.assign(document.createElement('div'), { className: 'thumb-box' });
    const ph = Object.assign(document.createElement('span'), { className: 'thumb-ph', textContent: i });
    box.append(ph);
    el.append(box, Object.assign(document.createElement('span'), { className: 'thumb-num', textContent: i }));
    el.addEventListener('click', () => { handle?.goTo(i); updateThumbsActive(i); });
    items.push({ num: i, el, box, ph, canvas: null, want: false, busy: false, tries: 0 });
    list.append(el);
    io.observe(el);
  }
  updateThumbsActive(start);
  if (pinned) toggleThumbs(true); else tab.hidden = false;
}
export function destroyThumbs() {
  io?.disconnect();
  io = null;
  handle = null;
  items.length = 0;
  active = null;
  list.replaceChildren();
  panel.hidden = true;
  tab.hidden = true;
  btn.classList.remove('active');
  popBack?.();
  popBack = null;
}
btn.addEventListener('click', () => toggleThumbs());
tab.addEventListener('click', () => toggleThumbs(true));
document.getElementById('thumbs-close').addEventListener('click', () => toggleThumbs(false));
const paintPin = () => pinBtn.classList.toggle('on', pinned);
pinBtn.addEventListener('click', () => { pinned = !pinned; setPref('thumbspin', pinned); paintPin(); });
paintPin();
// A page that was just drawn is the cheapest thumbnail there is.
document.getElementById('viewer-pages').addEventListener('pageready', (e) => {
  const it = items[e.detail.i];
  if (it && !panel.hidden && !it.canvas) want(it, true);
});
