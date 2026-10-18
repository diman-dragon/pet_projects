// OCR sheet: pick recognition languages, recognize this page or the whole document, show progress (cancellable).
import { recognizePages } from '../app/ocr.js';
import { ocrInfo, stopOcr } from '../io/ocr-engine.js';
import { getActiveHandle, getActiveDoc } from './viewer-handle.js';
import { toast } from './toast.js';
import { t } from '../core/i18n.js';
import { on } from '../core/bus.js';
import { getPref, setPref } from '../core/prefs.js';
const $ = (id) => document.getElementById(id);
const sheet = $('ocr-sheet'), chips = $('ocr-langs'), run = $('ocr-run');
let page = 1, total = 1, busy = false, stop = false;
let langs = getPref('ocrlangs', ['eng', 'rus']);
async function paintChips() {
  const info = await ocrInfo();
  chips.replaceChildren(...(info?.langs ?? []).map((code) => {
    const b = Object.assign(document.createElement('button'), { className: `ocr-chip${langs.includes(code) ? ' on' : ''}`, textContent: t(`ocr.lang.${code}`) });
    b.addEventListener('click', () => {
      langs = langs.includes(code) ? langs.filter((x) => x !== code) : [...langs, code];
      if (!langs.length) langs = [code];
      setPref('ocrlangs', langs);
      paintChips();
    });
    return b;
  }));
}
export async function toggleOcr(force) {
  sheet.hidden = !(force ?? sheet.hidden);
  if (!sheet.hidden) { await paintChips(); if (!(await ocrInfo())) toast(t('ocr.missing')); }
}
const errBox = $('ocr-error');
function showError(msg) { errBox.textContent = msg; errBox.hidden = false; }
async function start(pages) {
  const h = getActiveHandle(), doc = getActiveDoc();
  if (busy || !h || !doc) return;
  busy = true; stop = false; run.hidden = false; errBox.hidden = true;
  const fill = $('ocr-fill'), status = $('ocr-status');
  try {
    const done = await recognizePages(h, doc.id, pages, langs, {
      cancelled: () => stop,
      onPage: (i, n, p) => { fill.style.width = `${((i - 1 + p) / n) * 100}%`; status.textContent = t('ocr.progress').replace('{i}', i).replace('{n}', n).replace('{p}', Math.round(p * 100)); },
    });
    if (!stop) { toast(t(done ? 'ocr.done' : 'ocr.nothing')); sheet.hidden = true; }
  } catch (err) {
    console.error('OCR failed', err);
    const missing = err.message === 'ocr-missing' || err.message === 'ocr-lang';
    showError(missing ? t('ocr.missing') : `${t('ocr.fail')}: ${String(err?.message ?? err).slice(0, 300)}`);
  }
  busy = false; run.hidden = true;
}
$('ocr-page').addEventListener('click', () => start([page - 1]));
$('ocr-all').addEventListener('click', () => start(Array.from({ length: total }, (_, i) => i)));
$('ocr-close').addEventListener('click', () => toggleOcr(false));
$('ocr-cancel').addEventListener('click', () => { stop = true; stopOcr(); });
on('pagechange', (p) => { page = p.n; total = p.total; });
on('viewerclose', () => { stop = true; sheet.hidden = true; stopOcr(); });
