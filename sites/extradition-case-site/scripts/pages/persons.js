import { store } from '../store.js';
import { escapeHtml, sanitizeRichText } from '../security.js';
import { createPageShell, appendPageSummary } from '../components/page-shell.js';
import { getPageAside } from '../components/page-asides.js';
import { createRecordRow, createSectionHeading, createStatsGrid, splitRichText } from '../components/record-layout.js';
import { createDocumentPreview } from '../components/docs/preview.js';
import { flattenCatalog, getFileType } from '../components/docs/catalog.js';
import { resolveDocMeta } from '../utils/resolve-i18n.js';

export async function renderPersonsPage(container) {
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

  const [personsResp, docI18n, catalog] = await Promise.all([
    fetch(`./scripts/data/i18n/persons/${lang}.json`).then(r => r.ok ? r : fetch('./scripts/data/i18n/persons/ru.json')),
    loadModularI18n(lang),
    loadModularCatalog()
  ]);

  const t = await personsResp.json();
  const aside = getPageAside('persons', lang);
  const ui = docI18n.ui || {};
  const allDocs = flattenCatalog(catalog, { ...ui, __i18n: docI18n });
  const preview = createDocumentPreview({ ui });
  container._docsPreview = preview;

  const { body, after } = createPageShell(container, {
    pageClass: 'persons-page',
    badge: lang === 'ru' ? 'Действующие лица' : lang === 'sr' ? 'Učesnici' : 'Actors',
    title: t.title,
    subtitle: t.subtitle,
    intro: t.intro,
    asideLabel: aside.label,
    asideText: aside.text,
  });


  body.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Контуры' : lang === 'sr' ? 'Konture' : 'Contours',
    title: t.clusters.title,
  }));

  const clusterList = document.createElement('section');
  t.clusters.items.forEach((item, index) => {
    clusterList.appendChild(createRecordRow({
      eyebrow: lang === 'ru' ? `Контур ${index + 1}` : lang === 'sr' ? `Kontura ${index + 1}` : `Contour ${index + 1}`,
      status: item.label,
      title: item.title,
      tone: item.tone || 'default',
      bodyHtml: `<p>${sanitizeRichText(item.desc)}</p>`,
    }));
  });
  body.appendChild(clusterList);

  body.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Фигуранты' : lang === 'sr' ? 'Akteri' : 'Actors',
    title: t.profiles.title,
  }));

  const profileList = document.createElement('section');
  profileList.className = 'persons-list';

  t.profiles.items.forEach((item, index) => {
    const linkedDocs = allDocs.filter(d => 
      (item.id && d.signatoryId === item.id) || 
      (item.docIds && item.docIds.includes(d.id))
    );

    let docsHtml = '';
    if (linkedDocs.length > 0) {
      docsHtml = `
        <div class="persons-docs-link">
          <p class="persons-docs-label">${escapeHtml(t.labels?.doc ?? 'Documents')}:</p>
          <div class="persons-docs-grid">
            ${linkedDocs.map(doc => {
              const meta = resolveDocMeta(docI18n, doc.title_i18n_key);
              return `
                <button type="button" class="persons-doc-btn" data-doc-id="${doc.id}">
                  <span class="persons-doc-btn__icon">📄</span>
                  <span class="persons-doc-btn__text">${escapeHtml(meta.title)}</span>
                </button>
              `;
            }).join('')}
          </div>
        </div>
      `;
    } else {
      docsHtml = `<div class="record-focus"><span class="record-focus__label">${t.labels?.doc ?? 'Document'}</span>${splitRichText(item.doc)}</div>`;
    }

    profileList.appendChild(createRecordRow({
      eyebrow: lang === 'ru' ? `Фигура ${index + 1}` : lang === 'sr' ? `Akter ${index + 1}` : `Actor ${index + 1}`,
      status: t.labels?.role ?? 'Role',
      title: item.name,
      tone: item.tone || 'danger',
      bodyHtml: `
        <div class="persons-profile-header">
          ${item.photo ? `
            <div class="persons-profile-photo">
              <img src="${escapeHtml(item.photo)}" 
                   alt="${escapeHtml(item.name)}"
                   onerror="this.onerror=null;this.src='https://ui-avatars.com/api/?name=${encodeURIComponent(item.name)}&background=777&color=fff&size=128'">
            </div>` : ''}
          <div class="persons-profile-info">
            <div class="record-focus"><span class="record-focus__label">${t.labels?.role ?? 'Role'}</span>${splitRichText(item.role)}</div>
          </div>
        </div>
        ${docsHtml}
        <div class="record-focus"><span class="record-focus__label">${t.labels?.action ?? 'Action'}</span>${splitRichText(item.action)}</div>
      `,
    }));
  });

  profileList.querySelectorAll('.persons-doc-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const docId = btn.dataset.docId;
      const doc = allDocs.find(d => d.id === docId);
      if (doc) {
        const meta = resolveDocMeta(docI18n, doc.title_i18n_key);
        preview.openPreview(doc, meta, getFileType(doc.filename));
      }
    });
  });

  body.appendChild(profileList);

  appendPageSummary(after, t.summary, 'danger');
}
