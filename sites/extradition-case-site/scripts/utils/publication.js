import { escapeHtml, safeUrl } from '../security.js';

let logosCache = null;

export async function getPublicationLogos() {
  if (!logosCache) {
    const r = await fetch('./scripts/data/media-logos.json');
    logosCache = await r.json();
  }
  return logosCache;
}

/** @param {{ logo?: string, logo_url?: string, logo_alt?: string, source?: string }} item */
export function publicationLogoHtml(item, logos) {
  const url = item.logo_url || (item.logo && logos?.[item.logo]);
  if (url && safeUrl(url) !== '#') {
    return `<div class="pub-logo-wrap"><img class="pub-logo" src="${safeUrl(url)}" alt="${escapeHtml(item.logo_alt || item.source || '')}" width="150" height="30" loading="lazy"></div>`;
  }
  if (item.source) {
    return `<div class="pub-logo-wrap"><span class="pub-source">${escapeHtml(item.source)}</span></div>`;
  }
  return '';
}

export function sortByDateDesc(items, key = 'sort') {
  return [...items].sort((a, b) => (b[key] || '').localeCompare(a[key] || ''));
}
