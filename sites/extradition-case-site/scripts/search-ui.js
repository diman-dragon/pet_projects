/**
 * search-ui.js — renders search results into the main container.
 * Depends on the search index (search.js) and display helpers.
 */

import { store } from './store.js';
import { buildSearchIndex, searchInIndex } from './search.js';
import { escapeHtml } from './security.js';
import { highlightSnippet } from './highlight.js';

const SITE_NAME = 'Extradition Case';
const MAX_TERM_LENGTH = 100;

const I18N = {
  ru: {
    title: 'Поиск',
    found: (n) => `Найдено на ${n} ${n === 1 ? 'странице' : 'страницах'}`,
    none: 'Ничего не найдено.',
    go: 'Перейти',
  },
  en: {
    title: 'Search',
    found: (n) => `Found on ${n} ${n === 1 ? 'page' : 'pages'}`,
    none: 'Nothing found.',
    go: 'Go to',
  },
  sr: {
    title: 'Pretraga',
    found: (n) => `Pronađeno na ${n} ${n === 1 ? 'stranici' : n >= 2 && n <= 4 ? 'stranice' : 'stranica'}`,
    none: 'Ništa nije pronađeno.',
    go: 'Idi na',
  },
};

let searchIndex = null;

export function invalidateSearchIndex() {
  searchIndex = null;
}

async function getSearchIndex() {
  if (!searchIndex) {
    searchIndex = await buildSearchIndex(store.state.lang);
  }
  return searchIndex;
}

function buildEmptyResult(page, s, safeTerm) {
  const h2 = document.createElement('h2');
  h2.textContent = `${s.title}: «${safeTerm}»`;
  const p = document.createElement('p');
  p.style.cssText = 'color: var(--text-muted); margin-top: 1rem;';
  p.textContent = s.none;
  page.appendChild(h2);
  page.appendChild(p);
}

function buildResultCard(pageId, pageTitle, snippets, safeTerm, goLabel, onNavigate) {
  const card = document.createElement('div');
  card.style.cssText = 'background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.25rem;';

  const header = document.createElement('div');
  header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;';

  const strong = document.createElement('strong');
  strong.style.fontSize = '1rem';
  strong.textContent = pageTitle;

  const link = document.createElement('a');
  link.href = 'javascript:void(0)';
  link.dataset.page = pageId;
  link.style.cssText = 'font-size:0.85rem;color:var(--accent);text-decoration:none;';
  link.textContent = `${goLabel} →`;
  link.addEventListener('click', () => onNavigate(pageId));

  header.appendChild(strong);
  header.appendChild(link);
  card.appendChild(header);

  const snippetContainer = document.createElement('div');
  snippetContainer.style.cssText = 'display:flex;flex-direction:column;gap:0.5rem;';
  snippets.forEach(snippet => {
    const div = document.createElement('div');
    div.style.cssText = 'font-size:0.9rem;padding:0.5rem 0.75rem;background:var(--surface-strong);border-left:3px solid var(--accent);border-radius:0 4px 4px 0;line-height:1.5;';
    div.innerHTML = highlightSnippet(escapeHtml(snippet), safeTerm);
    snippetContainer.appendChild(div);
  });

  card.appendChild(snippetContainer);
  return card;
}

/**
 * Render search results into `container` for the given `term`.
 * `onNavigate(pageId)` is called when the user clicks a result link.
 */
export async function renderSearchResults(container, term, onNavigate) {
  const safeTerm = term.slice(0, MAX_TERM_LENGTH);
  const lang = store.state.lang;
  const s = I18N[lang] || I18N.en;

  document.title = `${s.title}: «${safeTerm}» — ${SITE_NAME}`;

  const index = await getSearchIndex();
  const results = searchInIndex(index, safeTerm);

  container.innerHTML = '';
  const page = document.createElement('div');
  page.className = 'page';

  if (results.length === 0) {
    buildEmptyResult(page, s, safeTerm);
  } else {
    const h2 = document.createElement('h2');
    h2.textContent = `${s.title}: «${safeTerm}»`;
    const summary = document.createElement('p');
    summary.style.cssText = 'color: var(--text-muted); margin-top: 0.25rem; margin-bottom: 1.5rem;';
    summary.textContent = s.found(results.length);
    page.appendChild(h2);
    page.appendChild(summary);

    const list = document.createElement('div');
    list.style.cssText = 'display:flex;flex-direction:column;gap:1.5rem;';
    results.forEach(({ page: pageId, pageTitle, snippets }) => {
      list.appendChild(buildResultCard(pageId, pageTitle, snippets, safeTerm, s.go, onNavigate));
    });
    page.appendChild(list);
  }

  container.appendChild(page);
}
