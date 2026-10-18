// One library card (cover + name + info); the same markup serves the grid and the list layout.
import { titleOf } from '../formats/registry.js';
import { icon } from './icon.js';
import { t } from '../core/i18n.js';
import { fmtSize } from '../core/format.js';
const hue = (s) => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7);
const ext = (name) => (name.includes('.') ? name.split('.').pop().slice(0, 4).toUpperCase() : '');
export function docCard(d) {
  const li = document.createElement('li');
  li.className = 'doc';
  li.dataset.id = d.id;
  li.dataset.src = d.src ?? 'local';
  li.style.setProperty('--hue', hue(d.name));
  const pct = d.pages > 0 ? Math.min(100, Math.round(((d.lastPage ?? 1) / d.pages) * 100)) : 0;
  li.innerHTML = `<button class="doc-main"><div class="cover"><div class="cover-ph">${icon('file', 26)}<b></b></div><img alt="" hidden><i class="prog"></i></div>`
    + '<div class="doc-info"><div class="doc-name"></div><div class="small muted doc-meta"></div></div></button>'
    + (d.src === 'folder' ? '' : `<button class="icon-btn doc-del">${icon('trash')}</button>`);
  li.querySelector('.cover-ph b').textContent = ext(d.name);
  li.querySelector('.prog').style.width = pct + '%';
  li.querySelector('.doc-name').textContent = titleOf(d.name);
  li.querySelector('.doc-meta').textContent = [fmtSize(d.size), d.pages ? `${d.lastPage ?? 1}/${d.pages}` : ''].filter(Boolean).join(' · ');
  li.querySelector('.doc-del')?.setAttribute('aria-label', t('home.delete'));
  return li;
}
