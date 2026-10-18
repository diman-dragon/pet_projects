import { store } from '../store.js';
import { escapeHtml } from '../security.js';
import { createPageShell } from '../components/page-shell.js';
import { getPageAside } from '../components/page-asides.js';
import { createSectionHeading } from '../components/record-layout.js';
import { createDocumentCard } from '../components/docs/document-card.js';
import { createDocumentPreview } from '../components/docs/preview.js';
import { flattenCatalog, getFileType, groupRelatedDocs } from '../components/docs/catalog.js';
import { resolveDocMeta, resolveI18n } from '../utils/resolve-i18n.js';
import {
  langLabelFactory,
  createArchiveMap,
  createFilterSelect,
  createSectionBrief,
  getFilterOptions
} from './docs/docs-ui.js';
import {
  normalizeDate,
  filterDocuments
} from './docs/docs-data.js';
import '../components/site-search.js';

export async function renderDocumentsPage(container) {
  const previousPreview = container._docsPreview;
  if (previousPreview?.destroy) previousPreview.destroy();

  const lang = store.state.lang;

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

  async function loadModularCatalog() {
    const index = await fetch('./scripts/data/documents.json').then(r => r.json());
    const categories = await Promise.all(
      index.categoryFiles.map(f => fetch(`./scripts/data/catalog/${f}`).then(r => r.json()))
    );
    return { categories };
  }

  const [i18n, catalog, navResp] = await Promise.all([
    loadModularI18n(lang),
    loadModularCatalog(),
    fetch(`./scripts/data/i18n/nav/${lang}.json`).then((r) => (r.ok ? r : fetch('./scripts/data/i18n/nav/ru.json'))),
  ]);

  const nav = await navResp.json();
  const ui = i18n.ui || {};
  ui.__i18n = i18n;

  const allDocs = flattenCatalog(catalog, ui);
  const aside = getPageAside('docs', lang);
  const preview = createDocumentPreview({ ui });
  const langLabel = langLabelFactory(i18n);
  const filterOptions = getFilterOptions(ui);
  container._docsPreview = preview;

  let activeCategory = catalog.categories[0]?.id || '';
  let activeSub = null;
  let searchTerm = '';
  let activeType = 'all';
  let activeInstitution = 'all';
  let activeVariant = 'all';

  const { body, after } = createPageShell(container, {
    pageClass: 'docs-page',
    badge: lang === 'ru' ? 'Архив документов' : lang === 'sr' ? 'Arhiva dokumenata' : 'Document archive',
    title: i18n.title,
    subtitle: i18n.subtitle,
    asideLabel: aside.label,
    asideText: aside.text,
  });

  after.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Навигация по архиву' : lang === 'sr' ? 'Kretanje kroz arhivu' : 'Archive navigation',
    title: lang === 'ru'
      ? 'Документы сгруппированы по смыслу и процессу'
      : lang === 'sr'
        ? 'Dokumenti su složeni po smislu i toku postupka'
        : 'Documents are grouped by logic and procedural track',
    text: lang === 'ru'
      ? 'Сначала выберите раздел слева, затем уточните блок или воспользуйтесь поиском и фильтрами.'
      : lang === 'sr'
        ? 'Najpre izaberite odeljak levo, zatim suzite blok ili koristite pretragu i filtere.'
        : 'Start with a section on the left, then narrow to a block or use search and filters.',
  }));

  body.innerHTML += `
    <div class="docs-layout">
      <nav class="docs-nav" id="docs-primary-nav" aria-label="${escapeHtml(ui.navPrimary || 'Categories')}"></nav>
      <div class="docs-main">
        <div class="docs-toolbar">
          <div class="docs-search-card">
            <p class="docs-search-card__eyebrow">${escapeHtml(ui.searchTitle || 'Search the archive')}</p>
            <p class="docs-search-card__hint">${escapeHtml(ui.searchHint || '')}</p>
            <site-search id="doc-search-comp" placeholder="${escapeHtml(nav.search || ui.searchPlaceholder || '')}"></site-search>
          </div>
          <div class="docs-filters-card">
            <p class="docs-search-card__eyebrow">${escapeHtml(ui.filtersTitle || 'Filters')}</p>
            <p class="docs-search-card__hint">${escapeHtml(ui.filtersHint || '')}</p>
            <div class="docs-toolbar__filters">
              ${createFilterSelect('docs-type-filter', ui.filterType || 'Document type', filterOptions.types)}
              ${createFilterSelect('docs-institution-filter', ui.filterInstitution || 'Institution / track', filterOptions.institutions)}
              ${createFilterSelect('docs-variant-filter', ui.filterVariant || 'Original / translation', filterOptions.variants)}
            </div>
          </div>
        </div>
        <div class="docs-route-map" id="docs-route-map"></div>
        <div class="docs-subnav" id="docs-subnav" hidden></div>
        <div class="docs-panel" id="docs-panel"></div>
      </div>
    </div>
  `;

  const primaryNav = body.querySelector('#docs-primary-nav');
  const routeMap = body.querySelector('#docs-route-map');
  const subNav = body.querySelector('#docs-subnav');
  const panel = body.querySelector('#docs-panel');

  function getCategory(catId) {
    return catalog.categories.find((cat) => cat.id === catId);
  }

  function renderPrimaryNav() {
    primaryNav.innerHTML = '';
    catalog.categories.forEach((cat) => {
      const label = resolveI18n(i18n, cat.title_i18n_key) || cat.id;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'docs-nav__btn';
      btn.dataset.cat = cat.id;
      btn.textContent = typeof label === 'string' ? label : label?.title || cat.id;
      btn.setAttribute('aria-current', cat.id === activeCategory ? 'true' : 'false');
      btn.addEventListener('click', () => {
        activeCategory = cat.id;
        activeSub = null;
        render();
      });
      primaryNav.appendChild(btn);
    });
  }

  function renderRouteMap() {
    if (!routeMap) return;
    const map = createArchiveMap({
      categories: catalog.categories,
      i18n,
      ui,
      activeCategory,
      onSelect: (catId) => {
        activeCategory = catId;
        activeSub = getCategory(catId)?.subcategories?.[0]?.id || null;
        render();
      },
    });
    routeMap.replaceChildren(map);
  }

  function renderSubNav(cat) {
    subNav.innerHTML = '';
    if (!cat?.subcategories?.length) {
      subNav.hidden = true;
      return;
    }

    subNav.hidden = false;
    if (!activeSub || !cat.subcategories.some((sub) => sub.id === activeSub)) {
      activeSub = cat.subcategories[0].id;
    }

    cat.subcategories.forEach((sub) => {
      const label = resolveI18n(i18n, sub.title_i18n_key) || sub.id;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'docs-subnav__btn';
      btn.textContent = typeof label === 'string' ? label : label?.title || sub.id;
      btn.setAttribute('aria-current', sub.id === activeSub ? 'true' : 'false');
      btn.addEventListener('click', () => {
        activeSub = sub.id;
        render();
      });
      subNav.appendChild(btn);
    });
  }

  function appendCards(fragment, docs) {
    const { byGroup, ungrouped } = groupRelatedDocs(docs, i18n);
    const variantOrder = { original: 0, translation: 1, serbian: 2 };

    // Build a map of all documents by ID for quick lookup
    const docMap = new Map();
    docs.forEach(doc => docMap.set(doc.id, doc));

    // Identify threads
    const childrenByParent = new Map();
    const roots = [];

    docs.forEach(doc => {
      if (doc.threadParentId && docMap.has(doc.threadParentId)) {
        if (!childrenByParent.has(doc.threadParentId)) {
          childrenByParent.set(doc.threadParentId, []);
        }
        childrenByParent.get(doc.threadParentId).push(doc);
      } else {
        roots.push(doc);
      }
    });

    const items = [];

    roots.forEach((doc) => {
      const meta = resolveDocMeta(i18n, doc.title_i18n_key);
      const groupDocs = meta.group ? byGroup.get(meta.group) || [] : [];
      const relatedToDoc = groupDocs.filter(d => d.id !== doc.id && !d.threadParentId);

      items.push({
        sortDate: normalizeDate(meta.date),
        variantRank: variantOrder[doc.variant || 'original'] || 0,
        render: () => {
          fragment.appendChild(createDocumentCard({
            docRow: doc,
            i18n,
            lang,
            ui,
            related: relatedToDoc,
            langLabel,
            onPreview: preview.openPreview,
            onRelatedNavigate: (id) => {
              const target = allDocs.find((item) => item.id === id);
              if (!target) return;
              activeCategory = target.categoryId;
              activeSub = target.subcategoryId;
              render();
              preview.openPreview(target, resolveDocMeta(i18n, target.title_i18n_key), getFileType(target.filename));
            },
          }));

          function renderChildren(parentId) {
            const children = childrenByParent.get(parentId) || [];
            children.sort((a, b) => normalizeDate(resolveDocMeta(i18n, a.title_i18n_key).date).localeCompare(normalizeDate(resolveDocMeta(i18n, b.title_i18n_key).date)));
            children.forEach(child => {
              const childMeta = resolveDocMeta(i18n, child.title_i18n_key);
              const childGroupDocs = childMeta.group ? byGroup.get(childMeta.group) || [] : [];
              const relatedToChild = childGroupDocs.filter(d => d.id !== child.id && !d.threadParentId);

            fragment.appendChild(createDocumentCard({
              docRow: child,
              i18n,
              lang,
              ui,
              threaded: true,
              related: relatedToChild,
              langLabel,
              onPreview: preview.openPreview,
              onRelatedNavigate: (id) => {
                const target = allDocs.find((item) => item.id === id);
                if (!target) return;
                activeCategory = target.categoryId;
                activeSub = target.subcategoryId;
                render();
                preview.openPreview(target, resolveDocMeta(i18n, target.title_i18n_key), getFileType(target.filename));
              },
            }));
              renderChildren(child.id);
            });
          }

          renderChildren(doc.id);
        },
      });
    });

    items.sort((a, b) => {
      const dateDiff = b.sortDate.localeCompare(a.sortDate);
      return dateDiff !== 0 ? dateDiff : a.variantRank - b.variantRank;
    });
    items.forEach((item) => item.render());
  }

  function renderSearchResults() {
    const term = searchTerm.trim().toLowerCase();
    const filters = { activeType, activeInstitution, activeVariant };
    const docs = filterDocuments(allDocs, filters).filter((doc) => {
      const meta = resolveDocMeta(i18n, doc.title_i18n_key);
      return [meta.title, meta.desc, meta.date].join(' ').toLowerCase().includes(term);
    });

    if (!docs.length) {
      panel.innerHTML = `<p class="docs-empty">${escapeHtml(ui.none)}</p>`;
      return;
    }

    panel.innerHTML = '';
    const frag = document.createDocumentFragment();
    const resultHeader = document.createElement('p');
    resultHeader.className = 'docs-empty docs-empty--inline';
    resultHeader.textContent = ui.searchResults?.replace('{n}', docs.length) || `${docs.length}`;
    frag.appendChild(resultHeader);
    appendCards(frag, docs);
    panel.appendChild(frag);
  }

  function renderCategoryList(cat) {
    panel.innerHTML = '';
    const frag = document.createDocumentFragment();
    const filters = { activeType, activeInstitution, activeVariant };

    const catMeta = resolveI18n(i18n, cat.title_i18n_key);
    const catBrief = createSectionBrief(catMeta, ui);
    if (catBrief) frag.appendChild(catBrief);

    const topDocs = filterDocuments(allDocs.filter((doc) => doc.categoryId === cat.id && !doc.subcategoryId), filters);
    if (topDocs.length) {
      appendCards(frag, topDocs);
    }

    if (cat.subcategories?.length) {
      const subs = activeSub ? cat.subcategories.filter((sub) => sub.id === activeSub) : cat.subcategories;
      subs.forEach((sub) => {
        const subMeta = resolveI18n(i18n, sub.title_i18n_key);
        const heading = document.createElement('h3');
        heading.className = 'docs-group__title';
        heading.textContent = typeof subMeta === 'string' ? subMeta : subMeta?.title || sub.id;
        frag.appendChild(heading);

        const subBrief = createSectionBrief(subMeta, ui);
        if (subBrief) frag.appendChild(subBrief);

        const docs = filterDocuments(allDocs.filter((doc) => doc.categoryId === cat.id && doc.subcategoryId === sub.id), filters);
        if (!docs.length) {
          const empty = document.createElement('p');
          empty.className = 'docs-empty';
          empty.textContent = ui.emptySection || ui.none;
          frag.appendChild(empty);
          return;
        }

        appendCards(frag, docs);
      });
    } else if (!cat.documents?.length) {
      const empty = document.createElement('p');
      empty.className = 'docs-empty';
      empty.textContent = ui.emptySection || ui.none;
      frag.appendChild(empty);
    }

    panel.appendChild(frag);
  }

  function render() {
    renderPrimaryNav();

    if (searchTerm.trim()) {
      subNav.hidden = true;
      renderSearchResults();
      return;
    }

    const cat = getCategory(activeCategory);
    renderSubNav(cat);
    if (!cat) {
      panel.innerHTML = `<p class="docs-empty">${escapeHtml(ui.none)}</p>`;
      return;
    }

    renderCategoryList(cat);
  }

  body.querySelector('#doc-search-comp')?.addEventListener('search', (event) => {
    searchTerm = event.detail;
    render();
  });

  body.querySelector('#docs-type-filter')?.addEventListener('change', (event) => {
    activeType = event.target.value;
    render();
  });
  body.querySelector('#docs-institution-filter')?.addEventListener('change', (event) => {
    activeInstitution = event.target.value;
    render();
  });
  body.querySelector('#docs-variant-filter')?.addEventListener('change', (event) => {
    activeVariant = event.target.value;
    render();
  });

  render();
}

