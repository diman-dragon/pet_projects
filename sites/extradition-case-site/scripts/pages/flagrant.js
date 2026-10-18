import { store } from '../store.js';
import { escapeHtml, sanitizeRichText } from '../security.js';
import { createPageShell, appendPageSummary } from '../components/page-shell.js';
import { getPageAside } from '../components/page-asides.js';
import { createRecordRow, createSectionHeading, createStatsGrid, documentCardHtml, sourceListHtml, splitRichText } from '../components/record-layout.js';

function navigateTo(pageId) {
  const header = document.querySelector('site-header');
  if (!header) return;
  header.dispatchEvent(new CustomEvent('navigate', {
    detail: pageId,
    bubbles: true,
    composed: true,
  }));
}

function parseHtmlLinks(items = []) {
  return items.map((html) => {
    const template = document.createElement('template');
    template.innerHTML = sanitizeRichText(html);
    const link = template.content.querySelector('a');
    if (!link) {
      return { label: template.content.textContent?.trim() || '' };
    }
    return { href: link.getAttribute('href') || '', label: link.textContent?.trim() || '' };
  }).filter((item) => item.label);
}

function refsLabel(lang, type) {
  const map = {
    ru: { echr: 'Прецеденты ЕСПЧ', un: 'Стандарты ООН' },
    sr: { echr: 'Praksa ESLJP', un: 'Standardi UN' },
    en: { echr: 'ECtHR authorities', un: 'UN standards' },
  };
  return map[lang]?.[type] || map.en[type];
}

function relevanceLabel(lang) {
  if (lang === 'ru') return 'Применимость';
  if (lang === 'sr') return 'Primenljivost';
  return 'Relevance';
}

function principlesHtml(block, lang) {
  if (!block.principles?.length) return '';
  return `
    <div class="record-episodes">
      ${block.principles.map((item) => `
        <article class="record-episode">
          <div class="record-episode__label">${escapeHtml(item.num)}</div>
          <div class="record-episode__body">
            <div class="record-quote">${escapeHtml(item.text)}</div>
            <div class="record-focus">
              <span class="record-focus__label">${relevanceLabel(lang)}</span>
              ${splitRichText(item.relevance)}
            </div>
          </div>
        </article>
      `).join('')}
    </div>
  `;
}

function episodeHtml(ep, t, lang) {
  return `
    <p>${splitRichText(ep.what_happened)}</p>
    ${ep.quote ? `<blockquote class="record-quote"><span class="record-quote__label">${escapeHtml(t.labels?.quote || '')}</span>${escapeHtml(ep.quote)}</blockquote>` : ''}
    ${ep.why_flagrant ? `
      <div class="record-focus">
        <span class="record-focus__label">${escapeHtml(t.labels?.why_flagrant || '')}</span>
        ${splitRichText(ep.why_flagrant)}
      </div>` : ''}
    ${ep.violations?.length ? `<div class="record-chips">${ep.violations.map((v) => `<span class="record-chip">${escapeHtml(v)}</span>`).join('')}</div>` : ''}
    ${ep.echr_refs_html?.length ? sourceListHtml(refsLabel(lang, 'echr'), parseHtmlLinks(ep.echr_refs_html), 'default') : ''}
    ${ep.un_refs_html?.length ? sourceListHtml(refsLabel(lang, 'un'), parseHtmlLinks(ep.un_refs_html), 'info') : ''}
    ${ep.doc_link ? documentCardHtml(ep.doc_link, t.labels?.doc_link || 'Document') : ''}
  `;
}

export async function renderFlagrantPage(container) {
  const lang = store.state.lang;
  let response = await fetch(`./scripts/data/i18n/flagrant/${lang}.json`);
  if (!response.ok) response = await fetch('./scripts/data/i18n/flagrant/ru.json');
  const t = await response.json();
  const aside = getPageAside('flagrant', lang);

  const { body, after } = createPageShell(container, {
    pageClass: 'flagrant-page',
    badge: t.badge,
    title: t.title,
    subtitle: t.subtitle,
    intro: t.intro,
    asideLabel: aside.label,
    asideText: aside.text,
  });


  const standardList = document.createElement('section');
  body.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Стандарты' : lang === 'sr' ? 'Standardi' : 'Standards',
    title: lang === 'ru' ? 'Почему отказ в рассмотрении по существу важен сам по себе' : lang === 'sr' ? 'Zašto je odbijanje meritorne ocene važno samo po sebi' : 'Why refusal to review the merits matters on its own',
  }));

  if (t.un_block) {
    standardList.appendChild(createRecordRow({
      eyebrow: 'UN',
      status: t.un_block.title,
      tone: 'info',
      bodyHtml: `
        <p>${splitRichText(t.un_block.intro)}</p>
        ${documentCardHtml({ title: t.un_block.url_label, href: t.un_block.url }, lang === 'ru' ? 'Официальный источник' : lang === 'sr' ? 'Zvanični izvor' : 'Official source')}
        ${principlesHtml(t.un_block, lang)}
      `,
    }));
  }

  if (t.echr_block) {
    standardList.appendChild(createRecordRow({
      eyebrow: 'ECtHR',
      status: t.echr_block.title,
      tone: 'danger',
      bodyHtml: (t.echr_block.paras || []).map((para) => `<p>${sanitizeRichText(typeof para === 'string' ? para : para.text || '')}</p>`).join(''),
    }));
  }

  body.appendChild(standardList);

  if (t.docs_cta) {
    const cta = document.createElement('div');
    cta.className = 'record-card record-card--danger';
    cta.innerHTML = `
      <div class="record-card__body">
        <strong>${escapeHtml(t.docs_cta.title)}</strong>
        <p>${escapeHtml(t.docs_cta.desc)}</p>
        <p><a href="javascript:void(0)" id="go-to-denials-btn" class="text-link">${escapeHtml(t.docs_cta.btn)} &rarr;</a></p>
      </div>
    `;
    body.appendChild(cta);
  }

  if (t.pattern_title || t.pattern_desc) {
    body.appendChild(createRecordRow({
      eyebrow: lang === 'ru' ? 'Схема' : lang === 'sr' ? 'Obrazac' : 'Pattern',
      status: t.pattern_title || '',
      bodyHtml: `<p>${splitRichText(t.pattern_desc || '')}</p>`,
    }));
  }

  const episodes = document.createElement('section');
  body.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Эпизоды' : lang === 'sr' ? 'Epizode' : 'Episodes',
    title: lang === 'ru' ? 'Повторяющийся шаблон формальных отказов' : lang === 'sr' ? 'Ponavljajući obrazac formalnih odbijanja' : 'The recurring pattern of formal refusals',
  }));
  t.episodes?.forEach((episode, index) => {
    episodes.appendChild(createRecordRow({
      eyebrow: episode.tag || `${index + 1}`,
      status: lang === 'ru' ? `Эпизод ${index + 1}` : lang === 'sr' ? `Epizoda ${index + 1}` : `Episode ${index + 1}`,
      title: episode.title,
      tone: 'danger',
      bodyHtml: episodeHtml(episode, t, lang),
    }));
  });
  body.appendChild(episodes);

  if (t.conclusion_title) {
    body.appendChild(createRecordRow({
      eyebrow: lang === 'ru' ? 'Вывод' : lang === 'sr' ? 'Zaključak' : 'Conclusion',
      status: t.conclusion_title,
      tone: 'danger',
      bodyHtml: (t.conclusion_paras || []).map((para) => `<p>${splitRichText(para)}</p>`).join(''),
    }));
  }

  appendPageSummary(after, t.summary, 'danger');

  const ctaBtn = container.querySelector('#go-to-denials-btn');
  if (ctaBtn) {
    ctaBtn.addEventListener('click', () => {
      navigateTo('docs');
      setTimeout(() => {
        const buttons = [...document.querySelectorAll('.docs-nav__btn')];
        const target = buttons.find((btn) => {
          const text = btn.textContent.toLowerCase();
          return btn.dataset.cat === 'otkazy'
            || text.includes('flagrant')
            || text.includes('denial')
            || text.includes('refusal')
            || text.includes('odbij');
        });
        if (target) target.click();
      }, 400);
    });
  }
}
