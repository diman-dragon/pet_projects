import { store } from '../store.js';
import { escapeHtml, safeUrl } from '../security.js';
import { createPageShell } from '../components/page-shell.js';
import { getPageAside } from '../components/page-asides.js';
import { createSectionHeading } from '../components/record-layout.js';
import { getPublicationLogos, publicationLogoHtml } from '../utils/publication.js';

export async function renderMediaPage(container) {
  const lang = store.state.lang;
  let response = await fetch(`./scripts/data/i18n/media/${lang}.json`);
  if (!response.ok) response = await fetch('./scripts/data/i18n/media/ru.json');
  const t = await response.json();
  const logos = await getPublicationLogos();
  const aside = getPageAside('media', lang);

  const { page, body, after } = createPageShell(container, {
    pageClass: 'media-page',
    badge: lang === 'ru' ? 'Медиаархив' : lang === 'sr' ? 'Medijska arhiva' : 'Media archive',
    title: t.title ?? 'Media',
    intro: t.manifesto ?? '',
    asideLabel: aside.label,
    asideText: aside.text,
  });

  body.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Публикации' : lang === 'sr' ? 'Objave' : 'Coverage',
    title: lang === 'ru' ? 'Независимые редакции, которые зафиксировали ключевые повороты дела' : lang === 'sr' ? 'Nezavisne redakcije koje su zabeležile ključne preokrete predmeta' : 'Independent newsrooms that recorded the case’s turning points',
  }));
  body.innerHTML += `<section id="media-list"></section>`;
  after.innerHTML = `<section id="press-call" class="page-summary media-page__press"></section>`;

  const list = body.querySelector('#media-list');
  t.items.forEach((item, index) => {
    const articleHref = safeUrl(item.link);
    const hasLink = articleHref && articleHref !== '#';
    const openLabel = escapeHtml(t.labels?.open_link ?? 'Open');
    const focusLabel = escapeHtml(t.labels?.focus ?? 'Focus');

    const row = document.createElement('article');
    row.className = 'media-item';
    row.innerHTML = `
      <aside class="media-item__aside">
        <time class="media-item__date" datetime="${escapeHtml(item.sort || '')}">${escapeHtml(item.date)}</time>
        ${publicationLogoHtml(item, logos)}
      </aside>
      <div class="media-item__card">
        <ui-card id="media-card-${index}"></ui-card>
      </div>
    `;
    list.appendChild(row);

    const card = row.querySelector(`#media-card-${index}`);
    if (card?.setContent) {
      card.setContent({
        type: item.source || '',
        title: item.title,
        text: `${escapeHtml(item.summary)}<div class="card-stack">
          <div class="media-focus">
            <strong class="media-focus__title">${focusLabel}</strong>
            ${escapeHtml(item.focus)}
          </div>
          ${hasLink ? `<a href="${articleHref}" target="_blank" rel="noopener noreferrer" class="text-link">${openLabel} →</a>` : ''}
        </div>`
      }, lang);
    }
  });

  const pressSection = page.querySelector('#press-call');
  const thesisLabel = escapeHtml(
    t.press_call.thesis_label ||
    (lang === 'ru' ? 'Позиция для СМИ' : lang === 'sr' ? 'Stav za medije' : 'Press statement')
  );
  pressSection.innerHTML = `
    <div class="media-page__press-title">${escapeHtml(t.press_call.title)}</div>
    <p>${escapeHtml(t.press_call.text)}</p>
    ${t.press_call.thesis ? `
    <blockquote class="prose-quote">
      <strong>${thesisLabel}</strong>
      ${escapeHtml(t.press_call.thesis)}
    </blockquote>` : ''}
  `;
}
