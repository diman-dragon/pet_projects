import { store } from '../store.js';
import { escapeHtml, safeUrl } from '../security.js';
import { getPublicationLogos, publicationLogoHtml, sortByDateDesc } from '../utils/publication.js';

function wireNavLinks(root) {
  root.querySelectorAll('[data-nav-page]').forEach((el) => {
    if (el._navWired) return;
    el._navWired = true;
    el.addEventListener('click', (e) => {
      e.preventDefault();
      document.querySelector('site-header').dispatchEvent(
        new CustomEvent('navigate', { detail: el.dataset.navPage, bubbles: true, composed: true }),
      );
    });
  });
}

function makeNavLink(page, label) {
  return `<a href="javascript:void(0)" class="text-link" data-nav-page="${escapeHtml(page)}">${escapeHtml(label)} &rarr;</a>`;
}

export async function renderMainPage(container) {
  const lang = store.state.lang;

  let t = null;
  try {
    let response = await fetch(`./scripts/data/i18n/home/${lang}.json`);
    if (!response.ok) response = await fetch('./scripts/data/i18n/home/ru.json');
    if (!response.ok) throw new Error(`HTTP ${response.status}: не удалось загрузить локализацию`);
    t = await response.json();
  } catch (err) {
    console.error('Ошибка загрузки локализации home:', err);
    container.innerHTML = `<div class="page home-page"><p style="padding:2rem;color:red;">Ошибка загрузки данных страницы. Попробуйте обновить.</p></div>`;
    return;
  }

  // Защита от неполного JSON
  if (!t?.main?.anatomy || !t?.sidebar) {
    console.error('Неверная структура локализации home:', t);
    container.innerHTML = `<div class="page home-page"><p style="padding:2rem;color:red;">Ошибка структуры данных страницы.</p></div>`;
    return;
  }

  const logos = await getPublicationLogos();
  const news = sortByDateDesc(t.sidebar.news || []);

  const newsHtml = news.map((item) => {
    const href = safeUrl(item.link);
    let titleInner = escapeHtml(item.title);

    if (item.link === '#docs' || item.link?.startsWith('#')) {
      const page = (item.link || '#docs').replace('#', '') || 'docs';
      titleInner = `<a href="javascript:void(0)" class="text-link" data-nav-page="${escapeHtml(page)}">${escapeHtml(item.title)}</a>`;
    } else if (href !== '#') {
      titleInner = `<a href="${href}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.title)}</a>`;
    }

    return `
      <article class="news-feed__item">
        <time class="news-feed__date" datetime="${escapeHtml(item.sort || '')}">${escapeHtml(item.date)}</time>
        ${publicationLogoHtml(item, logos)}
        <h4 class="news-feed__title">${titleInner}</h4>
        <p class="news-feed__desc">${escapeHtml(item.desc)}</p>
      </article>
    `;
  }).join('');

  const heroActions = {
    ru: {
      primary: 'Правовая оценка',
      secondary: 'Открыть архив документов',
      inside: 'Что уже собрано',
    },
    en: {
      primary: 'Legal analysis',
      secondary: 'Open document archive',
      inside: 'What is already assembled',
    },
    sr: {
      primary: 'Pravna analiza',
      secondary: 'Otvori arhivu dokumenata',
      inside: 'Šta je već sabrano',
    },
  }[lang] || {
    primary: 'Legal analysis',
    secondary: 'Open document archive',
    inside: 'What is already assembled',
  };

  const cards = t.main.cards || {};

  container.innerHTML = `
    <div class="page home-page">
      <section class="home-hero">
        <div class="home-hero__main">
          <div class="home-hero__eyebrow">Extradition Case Archive</div>
          <h1 class="home-hero__title">${escapeHtml(t.main.anatomy.title ?? '')}</h1>
          <p class="home-hero__subtitle">${escapeHtml(t.main.anatomy.subtitle ?? '')}</p>
          <p class="home-hero__text">${escapeHtml(t.main.anatomy.text ?? '')}</p>
          ${t.main.anatomy.manifesto ? `<blockquote class="prose-quote home-hero__quote">${escapeHtml(t.main.anatomy.manifesto)}</blockquote>` : ''}
          <div class="home-hero__actions">
            <a href="javascript:void(0)" class="home-hero__action home-hero__action--primary" data-nav-page="legal">${escapeHtml(heroActions.primary)}</a>
            <a href="javascript:void(0)" class="home-hero__action" data-nav-page="docs">${escapeHtml(heroActions.secondary)}</a>
          </div>
        </div>

        <div class="home-hero__aside">
          <section class="home-panel">
            <div class="home-panel__label"><a href="javascript:void(0)" class="text-link" data-nav-page="docs">${escapeHtml(heroActions.inside)}</a></div>
            <h2 class="home-panel__title"><a href="javascript:void(0)" class="text-link" data-nav-page="docs">${escapeHtml(cards.archive?.title ?? '')}</a></h2>
            <p class="home-panel__text">${escapeHtml(cards.archive?.text ?? '')}</p>
            <ul class="home-panel__list">
              ${(cards.archive?.list ?? []).map((entry) => `<li>${escapeHtml(entry)}</li>`).join('')}
            </ul>
          </section>

          <section class="home-panel">
            <div class="home-panel__label">${escapeHtml(cards.flagrant?.title ?? '')}</div>
            <h2 class="home-panel__title">${escapeHtml(cards.legal?.question ?? '')}</h2>
            <p class="home-panel__text">${escapeHtml(cards.flagrant?.text ?? '')}</p>
          </section>
        </div>
      </section>

      <page-grid>
        <section slot="main" class="home-content">
          <div class="home-content__section">
            <div class="home-content__eyebrow">${escapeHtml(cards.legal?.title ?? '')}</div>
            <h2 class="home-content__title">${escapeHtml(cards.legal?.question ?? '')}</h2>
          </div>
          <section class="home-cards">
            <ui-card id="card-legal"></ui-card>
            <ui-card id="card-international"></ui-card>
            <ui-card id="card-actors"></ui-card>
            <ui-card id="card-archive"></ui-card>
            <ui-card id="card-flagrant"></ui-card>
          </section>
        </section>

        <aside slot="sidebar" class="ui-card home-sidebar">
          <h3 class="home-sidebar__title">${escapeHtml(t.sidebar.title ?? '')}</h3>
          ${t.sidebar.note ? `<p class="home-sidebar__note">${escapeHtml(t.sidebar.note)}</p>` : ''}
          <div class="news-feed">${newsHtml}</div>
          <div class="home-sidebar__cta">
            <p class="home-sidebar__cta-title">${escapeHtml(t.sidebar.subscribe?.title ?? '')}</p>
            <p class="home-sidebar__cta-text">${escapeHtml(t.sidebar.subscribe?.text ?? '')}</p>
          </div>
          <p style="margin-top:var(--space);">
            <a href="javascript:void(0)" class="text-link" data-nav-page="docs">${escapeHtml(t.sidebar.archive_link ?? '')} &rarr;</a>
          </p>
        </aside>
      </page-grid>
    </div>
  `;

  wireNavLinks(container);

  if (cards.legal) {
    document.getElementById('card-legal')?.setContent({
      type: cards.legal.title,
      title: cards.legal.question,
      text: `${escapeHtml(cards.legal.text)}<br><br>${makeNavLink('legal', cards.legal.link)}`,
    }, lang);
  }

  if (cards.international) {
    document.getElementById('card-international')?.setContent({
      type: cards.international.title,
      title: cards.international.link,
      text: `${escapeHtml(cards.international.text)}<br><br>${makeNavLink('intl', cards.international.link)}`,
    }, lang);
  }

  if (cards.actors) {
    document.getElementById('card-actors')?.setContent({
      type: cards.actors.title,
      title: cards.actors.link,
      text: `${escapeHtml(cards.actors.text)}<br><br>${makeNavLink('persons', cards.actors.link)}`,
    }, lang);
  }

  if (cards.archive) {
    document.getElementById('card-archive')?.setContent({
      type: cards.archive.title,
      title: t.sidebar.archive_link,
      text: `${escapeHtml(cards.archive.text)}<ul>${(cards.archive.list ?? []).map((entry) => `<li>${escapeHtml(entry)}</li>`).join('')}</ul><br>${makeNavLink('docs', t.sidebar.archive_link)}`,
    }, lang);
  }

  if (cards.flagrant) {
    document.getElementById('card-flagrant')?.setContent({
      type: cards.flagrant.title,
      title: cards.flagrant.link,
      text: `${escapeHtml(cards.flagrant.text)}<br><br>${makeNavLink('flagrant', cards.flagrant.link)}`,
    }, lang);
  }

  wireNavLinks(container);
}
