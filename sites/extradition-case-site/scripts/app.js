import './components/site-header.js?v=20260517-qa4';
import './components/site-footer.js?v=20260517-qa4';
import './components/cookie-banner.js?v=20260517-qa4';
import './components/ui-card.js?v=20260517-qa4';
import './components/page-grid.js?v=20260517-qa4';
import { store } from './store.js';
import { invalidateSearchIndex, renderSearchResults as _renderSearchResults } from './search-ui.js';
import { clearHighlights } from './highlight.js';

const container = document.getElementById('app-container');
const _moduleCache = new Map();

async function getRenderFn(page) {
  if (_moduleCache.has(page)) return _moduleCache.get(page);
  
  let fn;
  if (page === 'media')    fn = (await import('./pages/media.js?v=20260517-qa4')).renderMediaPage;
  else if (page === 'intl')     fn = (await import('./pages/intl.js?v=20260517-qa4')).renderInternationalPage;
  else if (page === 'timeline') fn = (await import('./pages/timeline.js?v=20260517-qa4')).renderTimelinePage;
  else if (page === 'legal')    fn = (await import('./pages/legal.js?v=20260517-qa4')).renderLegalPage;
  else if (page === 'persons')  fn = (await import('./pages/persons.js?v=20260517-qa4')).renderPersonsPage;
  else if (page === 'docs')     fn = (await import('./pages/docs.js?v=20260517-qa4')).renderDocumentsPage;
  else if (page === 'flagrant') fn = (await import('./pages/flagrant.js?v=20260517-qa4')).renderFlagrantPage;
  else if (page === 'article8') fn = (await import('./pages/article8.js?v=20260520-art8')).renderArticle8Page;
  else                          fn = (await import('./pages/home.js?v=20260517-qa4')).renderMainPage;
  
  _moduleCache.set(page, fn);
  return fn;
}

const PAGE_TITLES_I18N = {
  ru: { home: 'Главная', timeline: 'Хронология', legal: 'Правовая оценка', persons: 'Действующие лица', docs: 'Документы', intl: 'Международный контур', media: 'Медиа-архив', flagrant: 'Флагрантный отказ в правосудии', article8: 'Статья 8 ЕКПЧ: Семья' },
  en: { home: 'Overview', timeline: 'Timeline', legal: 'Legal Analysis', persons: 'Who\'s Who', docs: 'Documents', intl: 'International Proceedings', media: 'Press Coverage', flagrant: 'Flagrant denial of justice', article8: 'Article 8 ECHR: Family' },
  sr: { home: 'Pregled', timeline: 'Hronologija', legal: 'Pravna analiza', persons: 'Učesnici', docs: 'Dokumenti', intl: 'Međunarodni okvir', media: 'Medijska arhiva', flagrant: 'Flagrantno uskraćivanje pravde', article8: 'Član 8 EKLJP: Porodica' },
};

const SITE_NAME = 'Extradition Case';

function scrollToTop() {
  window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  if (document.scrollingElement) {
    document.scrollingElement.scrollTop = 0;
  }
}

function setDocumentTitle(page) {
  const lang = store.state.lang;
  const titles = PAGE_TITLES_I18N[lang] || PAGE_TITLES_I18N.en;
  const label = titles[page] || page;
  document.title = `${label} — ${SITE_NAME}`;
}

// ---------- Loading skeleton ----------

function showLoading() {
  container.innerHTML = `
    <div class="page page-loading" aria-live="polite" aria-busy="true">
      <div class="page-loading__spinner" aria-hidden="true"></div>
    </div>`;
}

// ---------- 404 ----------

function render404Page() {
  const lang = store.state.lang;
  const msg = {
    ru: { title: 'Страница не найдена', text: 'Запрошенная страница не существует.', btn: 'На главную' },
    en: { title: 'Page not found',       text: 'The requested page does not exist.',  btn: 'Go home' },
    sr: { title: 'Stranica nije nađena', text: 'Tražena stranica ne postoji.',         btn: 'Na početnu' },
  }[lang] || { title: 'Page not found', text: '', btn: 'Go home' };

  container.innerHTML = `
    <div class="page page-404">
      <h2>${msg.title}</h2>
      <p style="color:var(--text-muted);margin:1rem 0 2rem;">${msg.text}</p>
      <button class="btn-home-404" style="
        display:inline-flex;align-items:center;gap:0.5rem;
        background:var(--accent);color:var(--accent-soft);
        border:none;border-radius:999px;padding:0.7rem 1.5rem;
        font:inherit;font-size:var(--text-sm);font-weight:600;cursor:pointer;">
        ← ${msg.btn}
      </button>
    </div>`;
  container.querySelector('.btn-home-404').addEventListener('click', () => {
    store.setState({ activePage: 'home' });
    renderActivePage();
  });
  document.title = `${msg.title} — ${SITE_NAME}`;
}

// ---------- Page routing ----------

document.addEventListener('DOMContentLoaded', () => {
  const header = document.querySelector('site-header');
  let searchTimeout = null;

  const handleNavigate = (pageId) => {
    if (!pageId) return;
    store.setState({ activePage: pageId });
    renderActivePage();
  };

  header.addEventListener('search', (e) => {
    const term = e.detail.trim();
    clearTimeout(searchTimeout);
    if (term === '') {
      renderActivePage();
    } else {
      searchTimeout = setTimeout(() => {
        clearHighlights(container);
        _renderSearchResults(container, term, (pageId) => {
          store.setState({ activePage: pageId });
          renderActivePage();
        });
      }, 250);
    }
  });

  header.addEventListener('navigate', (e) => {
    handleNavigate(e.detail);
  });

  document.addEventListener('app:navigate', (e) => {
    handleNavigate(e.detail);
  });

  window.addEventListener('popstate', () => {
    store.syncFromUrl();
    renderActivePage();
  });
});

export async function renderActivePage() {
  clearHighlights(container);
  scrollToTop();

  const page = store.state.activePage;

  // GA4: track virtual page view for SPA navigation
  if (typeof gtag === 'function') {
    gtag('event', 'page_view', {
      page_title: document.title,
      page_location: window.location.href,
      page_path: '/' + (page === 'home' ? '' : page),
    });
  }

  if (page !== 'docs' && container._docsPreview?.destroy) {
    container._docsPreview.destroy();
    delete container._docsPreview;
  }
  setDocumentTitle(page);
  showLoading();

  try {
    const renderFn = await getRenderFn(page);
    await renderFn(container);
    scrollToTop();
  } catch (err) {
    console.error('Page render error:', err);
    render404Page();
    scrollToTop();
  }
}

// Apply initial settings synchronously before first render
document.documentElement.dataset.theme = store.state.theme;
document.documentElement.lang = store.state.lang;

let _prevLang = store.state.lang;

store.subscribe((state) => {
  document.documentElement.dataset.theme = state.theme;
  document.documentElement.lang = state.lang;
  if (state.lang !== _prevLang) {
    _prevLang = state.lang;
    invalidateSearchIndex();
    renderActivePage();
  }
});

renderActivePage();
