const pages = ['home', 'timeline', 'legal', 'persons', 'docs', 'intl', 'media', 'flagrant', 'article8'];

// i18n folder names differ from page IDs in some cases
const PAGE_I18N_FOLDER = {
  intl: 'international',
};

const _searchIndexCache = new Map();

/**
 * Recursively extract all string values from a JSON object.
 */
function extractStrings(obj, results = []) {
  if (typeof obj === 'string') {
    if (obj.length > 3) results.push(obj);
  } else if (Array.isArray(obj)) {
    for (let i = 0; i < obj.length; i++) extractStrings(obj[i], results);
  } else if (obj && typeof obj === 'object') {
    const keys = Object.keys(obj);
    for (let i = 0; i < keys.length; i++) extractStrings(obj[keys[i]], results);
  }
  return results;
}

/**
 * Build a flat search index from all page JSON files.
 * Returns array of { page, pageTitle, strings: string[] }
 */
export async function buildSearchIndex(lang) {
  if (_searchIndexCache.has(lang)) return _searchIndexCache.get(lang);

  let navTitles = {};
  try {
    let navResp = await fetch(`./scripts/data/i18n/nav/${lang}.json`);
    if (!navResp.ok && lang !== 'ru') navResp = await fetch(`./scripts/data/i18n/nav/ru.json`);
    if (navResp.ok) navTitles = await navResp.json();
  } catch (e) {}

  const index = [];
  const promises = pages.map(async (page) => {
    try {
      const folder = PAGE_I18N_FOLDER[page] || page;
      let response = await fetch(`./scripts/data/i18n/${folder}/${lang}.json`);
      if (!response.ok && lang !== 'ru') response = await fetch(`./scripts/data/i18n/${folder}/ru.json`);
      if (!response.ok) return;
      const data = await response.json();
      index.push({
        page,
        pageTitle: navTitles[page] || page,
        strings: extractStrings(data),
      });
    } catch (e) {
      console.error(`Error loading ${page} for search index:`, e);
    }
  });

  await Promise.all(promises);
  _searchIndexCache.set(lang, index);
  return index;
}

/**
 * Search the index for a term.
 * Returns array of { page, pageTitle, snippets: string[] }
 */
export function searchInIndex(index, term) {
  if (!term || term.length < 2) return [];
  const termLower = term.toLowerCase().trim();
  const results = [];

  for (let i = 0; i < index.length; i++) {
    const { page, pageTitle, strings } = index[i];
    const snippets = [];
    
    for (let j = 0; j < strings.length; j++) {
      const text = strings[j];
      const lower = text.toLowerCase();
      const idx = lower.indexOf(termLower);
      
      if (idx !== -1) {
        const start = Math.max(0, idx - 40);
        const end = Math.min(text.length, idx + termLower.length + 60);
        let snippet = text.slice(start, end).replace(/\s+/g, ' ').trim();
        
        if (start > 0) snippet = '\u2026' + snippet;
        if (end < text.length) snippet = snippet + '\u2026';
        
        // Basic de-duplication
        let exists = false;
        for (let k = 0; k < snippets.length; k++) {
          if (snippets[k].includes(snippet.slice(5, 15))) { exists = true; break; }
        }
        if (!exists) snippets.push(snippet);
        if (snippets.length >= 3) break;
      }
    }
    
    if (snippets.length > 0) {
      results.push({ page, pageTitle, snippets });
    }
  }

  return results;
}
