import { store } from '../store.js';
import { escapeHtml } from '../security.js';
import { createPageShell, appendPageSummary } from '../components/page-shell.js';
import { getPageAside } from '../components/page-asides.js';
import { createRecordRow, createSectionHeading, splitRichText } from '../components/record-layout.js';
import { createDocumentPreview } from '../components/docs/preview.js';
import { flattenCatalog, getFileType } from '../components/docs/catalog.js';
import { resolveDocMeta } from '../utils/resolve-i18n.js';

export async function renderTimelinePage(container) {
  const previousPreview = container._docsPreview;
  if (previousPreview?.destroy) previousPreview.destroy();

  const lang = store.state.lang;

  async function loadModularCatalog() {
    const index = await fetch('./scripts/data/documents.json').then(r => r.json());
    const categories = await Promise.all(
      index.categoryFiles.map(f => fetch(`./scripts/data/catalog/${f}`).then(r => r.json()))
    );
    return { categories };
  }

  async function loadModularI18n(targetLang) {
    const base = `./scripts/data/i18n/docs/${targetLang}/`;
    try {
      const [ui, categories, docs] = await Promise.all([
        fetch(`${base}ui.json`).then(r => r.json()),
        fetch(`${base}categories.json`).then(r => r.json()),
        fetch(`${base}documents.json`).then(r => r.json())
      ]);
      return { ...ui, category: categories, doc: docs };
    } catch (e) {
      if (targetLang !== 'ru') return loadModularI18n('ru');
      throw e;
    }
  }

  let response = await fetch(`./scripts/data/i18n/timeline/${lang}.json`);
  if (!response.ok) response = await fetch('./scripts/data/i18n/timeline/ru.json');
  const t = await response.json();
  const aside = getPageAside('timeline', lang);

  const [docI18n, catalog] = await Promise.all([
    loadModularI18n(lang),
    loadModularCatalog()
  ]);

  const ui = docI18n.ui || {};
  const allDocs = flattenCatalog(catalog, { ...ui, __i18n: docI18n });
  const preview = createDocumentPreview({ ui });
  container._docsPreview = preview;

  const { body, after } = createPageShell(container, {
    pageClass: 'timeline-page',
    badge: lang === 'ru' ? 'Хронология' : lang === 'sr' ? 'Hronologija' : 'Timeline',
    title: t.title,
    subtitle: t.subtitle,
    intro: t.intro,
    asideLabel: aside.label,
    asideText: aside.text,
  });

  body.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Линия времени' : lang === 'sr' ? 'Vremenska linija' : 'Timeline',
    title: lang === 'ru' ? 'События в последовательности, в которой они произошли' : lang === 'sr' ? 'Događaji redom kojim su se odvijali' : 'Events in the order they unfolded',
  }));

  const list = document.createElement('section');
  list.className = 'timeline-list';

  t.events.forEach((event) => {
    const doc = event.docId ? allDocs.find(d => d.id === event.docId) : null;
    let bodyHtml = `<p>${splitRichText(event.text)}</p>`;

    if (doc) {
      const meta = resolveDocMeta(docI18n, doc.title_i18n_key);
      bodyHtml += `
        <div class="timeline-doc-link">
          <button type="button" class="timeline-view-btn" data-doc-id="${doc.id}">
            <span class="timeline-view-btn__icon">📄</span>
            <span class="timeline-view-btn__text">${escapeHtml(ui.preview || 'View Document')}: ${escapeHtml(meta.title)}</span>
          </button>
        </div>
      `;
    }

    const row = createRecordRow({
      eyebrow: event.date,
      title: event.title,
      bodyHtml: bodyHtml,
    });
    list.appendChild(row);
  });

  list.querySelectorAll('.timeline-view-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const docId = btn.dataset.docId;
      const doc = allDocs.find(d => d.id === docId);
      if (doc) {
        const meta = resolveDocMeta(docI18n, doc.title_i18n_key);
        preview.openPreview(doc, meta, getFileType(doc.filename));
      }
    });
  });

  body.appendChild(list);
  appendPageSummary(after, t.summary);
}
