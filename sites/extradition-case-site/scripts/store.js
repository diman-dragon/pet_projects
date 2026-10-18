// Read persisted preferences on first load
const storedTheme = typeof localStorage !== 'undefined' ? localStorage.getItem('theme') : null;
const storedLang  = typeof localStorage !== 'undefined' ? localStorage.getItem('lang')  : null;

const VALID_PAGES = ['home', 'timeline', 'legal', 'persons', 'docs', 'intl', 'media', 'flagrant', 'article8'];
const VALID_LANGS = ['ru', 'sr', 'en'];

// Read initial page from URL hash, e.g. #flagrant → 'flagrant'
function getPageFromUrl() {
  const hash = window.location.hash.replace('#', '').trim();
  return VALID_PAGES.includes(hash) ? hash : 'home';
}

const state = {
  lang:       VALID_LANGS.includes(storedLang) ? storedLang : 'ru',
  theme:      storedTheme === 'light' ? 'light' : 'dark',
  searchTerm: '',
  activePage: getPageFromUrl(),
};
const subscribers = new Set();

function notify() {
  for (const fn of subscribers) {
    fn({ ...state });
  }
}

export const store = {
  get state() {
    return { ...state };
  },
  setState(patch) {
    Object.assign(state, patch);
    if (patch.theme) {
      localStorage.setItem('theme', state.theme);
    }
    if (patch.lang) {
      localStorage.setItem('lang', state.lang);
    }
    if (patch.activePage) {
      const hash = patch.activePage === 'home' ? '' : patch.activePage;
      const newUrl = hash ? `#${hash}` : window.location.pathname + window.location.search;
      if (window.location.hash.replace('#', '') !== hash) {
        history.pushState({ activePage: patch.activePage }, '', newUrl);
      }
    }
    notify();
  },
  subscribe(fn) {
    subscribers.add(fn);
    return () => subscribers.delete(fn);
  },
  syncFromUrl() {
    const hash = window.location.hash.replace('#', '').trim();
    state.activePage = VALID_PAGES.includes(hash) ? hash : 'home';
    notify();
  },
};
