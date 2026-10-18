import { store } from '../store.js';
import { escapeHtml } from '../security.js';
import { createPageShell, appendPageSummary } from '../components/page-shell.js';
import { getPageAside } from '../components/page-asides.js';
import { createRecordRow, createSectionHeading, sourceListHtml, documentCardHtml, splitRichText } from '../components/record-layout.js';

function refsLabel(lang) {
  if (lang === 'ru') return 'Источники';
  if (lang === 'sr') return 'Izvori';
  return 'Sources';
}

function docLabel(lang) {
  if (lang === 'ru') return 'Документ';
  if (lang === 'sr') return 'Dokument';
  return 'Document';
}

export async function renderInternationalPage(container) {
  const lang = store.state.lang;
  let response = await fetch(`./scripts/data/i18n/international/${lang}.json`);
  if (!response.ok) response = await fetch('./scripts/data/i18n/international/ru.json');
  const t = await response.json();
  const aside = getPageAside('intl', lang);

  const { body, after } = createPageShell(container, {
    badge: lang === 'ru' ? 'Международный контур' : lang === 'sr' ? 'Međunarodni okvir' : 'International track',
    title: t.title,
    subtitle: t.subtitle,
    intro: t.intro,
    asideLabel: aside.label,
    asideText: aside.text,
  });

  const list = document.createElement('section');
  body.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Инстанции' : lang === 'sr' ? 'Instance' : 'Institutions',
    title: lang === 'ru' ? 'Кто и что проверяет вне национального процесса' : lang === 'sr' ? 'Ko i šta proverava van nacionalnog postupka' : 'Who checks what beyond the national process',
  }));
  const focusLabel = escapeHtml(t.labels?.focus ?? 'Focus');

  t.items.forEach((item) => {
    const bodyHtml = `
      <p>${splitRichText(item.text)}</p>
      <div class="record-focus">
        <span class="record-focus__label">${focusLabel}</span>
        ${splitRichText(item.focus)}
      </div>
      ${item.sources?.length ? sourceListHtml(refsLabel(lang), item.sources) : ''}
      ${item.doc ? documentCardHtml(item.doc, docLabel(lang)) : ''}
    `;

    list.appendChild(createRecordRow({
      eyebrow: lang === 'ru' ? 'Инстанция' : lang === 'sr' ? 'Instanca' : 'Institution',
      status: item.status,
      title: item.org,
      tone: item.org?.includes('UN') ? 'info' : 'default',
      bodyHtml,
    }));
  });

  body.appendChild(list);
  appendPageSummary(after, t.summary);
}
