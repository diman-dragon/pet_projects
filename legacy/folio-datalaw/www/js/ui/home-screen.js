// Library screen: renders documents (grid or list) with lazy covers; open on tap, delete on long press / trash.
import { listDocs, removeDoc } from '../data/library.js';
import { openStored } from '../app/open-file.js';
import { docCard } from './doc-card.js';
import { watchCovers } from './cover-view.js';
import { renderShelf } from './reading-shelf.js';
import './home-toolbar.js';
import { toast } from './toast.js';
import { t } from '../core/i18n.js';
import { on } from '../core/bus.js';
const list = document.getElementById('doc-list');
const empty = document.getElementById('doc-empty');
const body = document.getElementById('home-body');
async function refresh() {
  const docs = await listDocs();
  const cards = docs.map((meta) => ({ meta, li: docCard(meta) }));
  list.replaceChildren(...cards.map((c) => c.li));
  empty.hidden = docs.length > 0;
  watchCovers(body, [...renderShelf(docs), ...cards]);
}
async function ask(li) {
  if (li.dataset.src === 'folder') return;
  if (confirm(t('home.deleteQ'))) { await removeDoc(li.dataset.id); refresh(); }
}
body.addEventListener('click', (e) => {
  const li = e.target.closest('.doc');
  if (!li) return;
  if (e.target.closest('.doc-del')) ask(li);
  else openStored(li.dataset.id).catch(() => toast(t('home.pickFail')));
});
list.addEventListener('contextmenu', (e) => {
  const li = e.target.closest('.doc');
  if (!li) return;
  e.preventDefault();
  ask(li);
});
on('libchange', refresh);
on('langchange', refresh);
refresh();
