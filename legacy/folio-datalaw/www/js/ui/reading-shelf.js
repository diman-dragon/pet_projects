// "Reading now" shelf above the library: a horizontal row of started documents with their progress.
import { readingDocs } from '../app/reading.js';
import { docCard } from './doc-card.js';
import { t } from '../core/i18n.js';
const shelf = document.getElementById('reading-shelf');
const row = document.getElementById('shelf-list');
export function renderShelf(docs) {
  const cards = readingDocs(docs).map((meta) => {
    const li = docCard(meta);
    li.querySelector('.doc-meta').textContent = meta.pages ? t('home.pageOf').replace('{n}', meta.lastPage).replace('{total}', meta.pages) : String(meta.lastPage);
    return { meta, li };
  });
  row.replaceChildren(...cards.map((c) => c.li));
  shelf.hidden = cards.length === 0;
  return cards;
}
