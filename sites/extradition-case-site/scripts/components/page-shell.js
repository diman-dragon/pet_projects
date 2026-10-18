import { escapeHtml } from '../security.js';

export function createPageShell(container, {
  pageClass = '',
  badge = '',
  title = '',
  subtitle = '',
  intro = '',
  asideLabel = '',
  asideText = '',
} = {}) {
  container.innerHTML = `
    <div class="page ${escapeHtml(pageClass).trim()}">
      <div class="page-shell">
        <header class="page-hero">
          <div class="page-hero__main">
            ${badge ? `<div class="page-hero__badge">${escapeHtml(badge)}</div>` : ''}
            <h2 class="page-hero__title">${escapeHtml(title)}</h2>
            ${subtitle ? `<p class="page-hero__subtitle">${escapeHtml(subtitle)}</p>` : ''}
            ${intro ? `<p class="page-hero__intro">${escapeHtml(intro)}</p>` : ''}
          </div>
          ${asideLabel || asideText ? `
            <aside class="page-hero__aside">
              ${asideLabel ? `<div class="page-hero__meta-label">${escapeHtml(asideLabel)}</div>` : ''}
              ${asideText ? `<p class="page-hero__meta-text">${escapeHtml(asideText)}</p>` : ''}
            </aside>
          ` : ''}
        </header>
        <div class="page-shell__body"></div>
        <div class="page-shell__after"></div>
      </div>
    </div>
  `;

  return {
    page: container.querySelector('.page'),
    body: container.querySelector('.page-shell__body'),
    after: container.querySelector('.page-shell__after'),
  };
}

export function appendPageSummary(target, text, tone = 'default') {
  if (!target || !text) return;
  const summary = document.createElement('section');
  summary.className = `page-summary page-summary--${tone}`;
  summary.innerHTML = `<p>${escapeHtml(text)}</p>`;
  target.appendChild(summary);
}
