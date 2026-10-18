import { store } from '../store.js';
import { createPageShell, appendPageSummary } from '../components/page-shell.js';
import { getPageAside } from '../components/page-asides.js';
import { createRecordRow, createSectionHeading, createStatsGrid, documentCardHtml, sourceListHtml, splitRichText } from '../components/record-layout.js';

function argumentLabel(lang, index) {
  if (lang === 'ru') return `Аргумент ${index + 1}`;
  if (lang === 'sr') return `Argument ${index + 1}`;
  return `Argument ${index + 1}`;
}

function thesisLabel(lang, index) {
  if (lang === 'ru') return `Тезис ${index + 1}`;
  if (lang === 'sr') return `Teza ${index + 1}`;
  return `Thesis ${index + 1}`;
}

function sourceLabel(lang) {
  if (lang === 'ru') return 'Источники';
  if (lang === 'sr') return 'Izvori';
  return 'Sources';
}

function episodeBlock(section, labels) {
  if (!section.highlight || !section.episodes?.length) return '';
  return `
    <div class="record-episodes">
      ${section.episodes.map((episode) => `
        <article class="record-episode">
          <div class="record-episode__label">${episode.label}</div>
          <div class="record-episode__body">
            <div class="record-focus">
              <span class="record-focus__label">${labels.content}</span>
              ${splitRichText(episode.what_protocol_says)}
            </div>
            <div class="record-focus">
              <span class="record-focus__label">${labels.summary}</span>
              ${splitRichText(episode.what_it_actually_means)}
            </div>
            ${episode.source ? `<div class="record-sources"><div class="record-sources__label">${sourceLabel(labels.lang)}</div><div class="record-sources__text">${splitRichText(episode.source)}</div></div>` : ''}
            ${episode.sources?.length ? sourceListHtml(sourceLabel(labels.lang), episode.sources) : ''}
          </div>
        </article>
      `).join('')}
    </div>
  `;
}

export async function renderLegalPage(container) {
  const lang = store.state.lang;
  let response = await fetch(`./scripts/data/i18n/legal/${lang}.json`);
  if (!response.ok) response = await fetch('./scripts/data/i18n/legal/ru.json');
  const t = await response.json();
  const aside = getPageAside('legal', lang);

  const { body, after } = createPageShell(container, {
    pageClass: 'flagrant-page',
    badge: lang === 'ru' ? 'Правовая оценка' : lang === 'sr' ? 'Pravna analiza' : 'Legal analysis',
    title: t.title,
    subtitle: t.subtitle,
    intro: t.intro,
    asideLabel: aside.label,
    asideText: aside.text,
  });


  const labels = {
    lang,
    content: t.labels?.content ?? 'Content',
    summary: t.labels?.summary ?? 'Summary',
  };

  if (t.reports?.length) {
    body.appendChild(createSectionHeading({
      kicker: lang === 'ru' ? 'Анонсы докладов' : lang === 'sr' ? 'Najave izveštaja' : 'Report announcements',
      title: t.reports_title,
    }));

    const reportsList = document.createElement('section');
    t.reports.forEach((report, index) => {
      reportsList.appendChild(createRecordRow({
        eyebrow: lang === 'ru' ? `Доклад ${index + 1}` : lang === 'sr' ? `Izveštaj ${index + 1}` : `Report ${index + 1}`,
        status: report.tag || '',
        title: '',
        tone: report.tone || 'danger',
        bodyHtml: `
          <div class="record-focus">
            <span class="record-focus__label">${lang === 'ru' ? 'Тема' : lang === 'sr' ? 'Tema' : 'Theme'}</span>
            ${splitRichText(report.title)}
          </div>
          <div class="record-focus">
            <span class="record-focus__label">${labels.summary}</span>
            ${splitRichText(report.summary)}
          </div>
        `,
      }));
    });
    body.appendChild(reportsList);
  }

  const sectionList = document.createElement('section');
  body.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Аргументы' : lang === 'sr' ? 'Argumenti' : 'Arguments',
    title: lang === 'ru' ? 'Где уголовная версия расходится с документами' : lang === 'sr' ? 'Gde se krivična verzija razilazi sa dokumentima' : 'Where the criminal narrative diverges from the record',
  }));
  t.sections.forEach((section, index) => {
    sectionList.appendChild(createRecordRow({
      eyebrow: argumentLabel(lang, index),
      status: section.highlight ? section.title : '',
      title: section.highlight ? '' : section.title,
      tone: section.tone || (section.highlight ? 'danger' : 'info'),
      bodyHtml: `
        <div class="record-focus">
          <span class="record-focus__label">${labels.content}</span>
          ${splitRichText(section.content)}
        </div>
        ${episodeBlock(section, labels)}
        <div class="record-focus">
          <span class="record-focus__label">${labels.summary}</span>
          ${splitRichText(section.summary)}
        </div>
        ${section.sources?.length ? sourceListHtml(sourceLabel(lang), section.sources) : ''}
      `,
    }));
  });
  body.appendChild(sectionList);

  if (t.theses?.length) {
    body.appendChild(createSectionHeading({
      kicker: lang === 'ru' ? 'Тезисы' : lang === 'sr' ? 'Teze' : 'Theses',
      title: t.theses_title,
    }));

    const thesisList = document.createElement('section');
    thesisList.className = 'record-stack';

    t.theses.forEach((thesis, index) => {
      thesisList.appendChild(createRecordRow({
        eyebrow: thesisLabel(lang, index),
        status: thesis.tag || '',
        title: thesis.title,
        tone: thesis.tone || 'danger',
        bodyHtml: `
          <p>${splitRichText(thesis.text)}</p>
          ${thesis.sources?.length ? sourceListHtml(sourceLabel(lang), thesis.sources) : ''}
          ${thesis.doc ? documentCardHtml(thesis.doc, lang === 'ru' ? 'Связанный документ' : lang === 'sr' ? 'Povezani dokument' : 'Related document') : ''}
        `,
      }));
    });

    body.appendChild(thesisList);
  }

  appendPageSummary(after, t.summary, 'danger');
}
