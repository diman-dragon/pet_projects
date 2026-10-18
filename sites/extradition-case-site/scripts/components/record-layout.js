import { escapeHtml, safeUrl } from '../security.js';

export function splitRichText(text) {
  return escapeHtml(text || '').replace(/\n/g, '<br>');
}

export function sourceListHtml(label, items = [], tone = 'default') {
  if (!items.length) return '';
  return `
    <div class="record-sources record-sources--${tone}">
      <div class="record-sources__label">${escapeHtml(label)}</div>
      <div class="record-sources__list">
        ${items.map((item) => {
          const href = safeUrl(item.href);
          const text = escapeHtml(item.label || item.href || '');
          return href === '#'
            ? `<span class="record-sources__text">${text}</span>`
            : `<a href="${href}" target="_blank" rel="noopener noreferrer" class="record-sources__link">${text}</a>`;
        }).join('')}
      </div>
    </div>
  `;
}

export function documentCardHtml(doc, eyebrow = 'Document') {
  if (!doc?.href && !doc?.file) return '';
  const href = safeUrl(doc.href || `./files/${doc.file}`);
  if (href === '#') return '';
  return `
    <a href="${href}" target="_blank" rel="noopener noreferrer" class="record-doc">
      <div class="record-doc__eyebrow">${escapeHtml(eyebrow)}</div>
      <div class="record-doc__title">${escapeHtml(doc.title || doc.label || doc.href)}</div>
      ${doc.ref ? `<div class="record-doc__meta">${escapeHtml(doc.ref)}</div>` : ''}
    </a>
  `;
}

export function createRecordRow({
  eyebrow = '',
  status = '',
  title = '',
  tag = '',
  tone = 'default',
  bodyHtml = '',
}) {
  const row = document.createElement('article');
  row.className = 'record-row';
  const resolvedTag = tag || status;
  row.innerHTML = `
    <aside class="record-row__aside">
      ${eyebrow ? `<div class="record-row__eyebrow">${escapeHtml(eyebrow)}</div>` : ''}
    </aside>
    <div class="record-card record-card--${escapeHtml(tone)}">
      ${resolvedTag ? `<div class="record-card__tag">${escapeHtml(resolvedTag)}</div>` : ''}
      ${title ? `<h3 class="record-card__title">${escapeHtml(title)}</h3>` : ''}
      <div class="record-card__body">${bodyHtml}</div>
    </div>
  `;
  return row;
}

export function createStatsGrid(items = []) {
  const grid = document.createElement('div');
  grid.className = 'stats-grid';
  items.forEach((item) => {
    const card = document.createElement('article');
    card.className = 'stat-card';
    card.innerHTML = `
      <div class="stat-card__value">${escapeHtml(item.value)}</div>
      <div class="stat-card__label">${escapeHtml(item.label)}</div>
    `;
    grid.appendChild(card);
  });
  return grid;
}

export function createSectionHeading({
  kicker = '',
  title = '',
  text = '',
} = {}) {
  const section = document.createElement('header');
  section.className = 'section-heading';
  section.innerHTML = `
    ${kicker ? `<div class="section-heading__kicker">${escapeHtml(kicker)}</div>` : ''}
    ${title ? `<h3 class="section-heading__title">${escapeHtml(title)}</h3>` : ''}
    ${text ? `<p class="section-heading__text">${escapeHtml(text)}</p>` : ''}
  `;
  return section;
}
