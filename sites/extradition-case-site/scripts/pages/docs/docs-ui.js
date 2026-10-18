import { escapeHtml } from '../../security.js';
import { resolveI18n } from '../../utils/resolve-i18n.js';

export function langLabelFactory(i18n) {
  return (code) => (i18n.lang_labels && i18n.lang_labels[code]) || code.toUpperCase();
}

export function createFilterSelect(id, label, options) {
  return `
    <label class="docs-filter-label" for="${escapeHtml(id)}">
      <span>${escapeHtml(label)}</span>
      <select id="${escapeHtml(id)}">
        ${options.map((option) => `
          <option value="${escapeHtml(option.value)}">${escapeHtml(option.label)}</option>
        `).join('')}
      </select>
    </label>
  `;
}

export function createArchiveMap({ categories, i18n, ui, activeCategory, onSelect }) {
  const map = document.createElement('section');
  map.className = 'docs-route';

  const steps = ui.archiveMapItems || [
    { categoryId: 'russia', label: 'РФ', text: 'Первичные доказательства, арбитраж и уголовное дело.' },
    { categoryId: 'serbia', label: 'Сербия', text: 'Экстрадиция, суды и азил в Сербии.' },
    { categoryId: 'international', label: 'Международный контур', text: 'Интерпол, ООН, ЕСПЧ и Украина.' },
  ].filter((item) => categories.some((cat) => cat.id === item.categoryId));

  map.innerHTML = `
    <div class="docs-route__head">
      <span>${escapeHtml(ui.archiveMapEyebrow || 'Case map')}</span>
      <strong>${escapeHtml(ui.archiveMapTitle || 'How the document archive is organized')}</strong>
    </div>
    <div class="docs-route__steps">
      ${steps.map((step, index) => {
        const cat = categories.find((item) => item.id === step.categoryId);
        const catMeta = cat ? resolveI18n(i18n, cat.title_i18n_key) : null;
        const title = step.label || (typeof catMeta === 'string' ? catMeta : catMeta?.title) || step.categoryId;
        return `
          <button type="button" class="docs-route__step" data-cat="${escapeHtml(step.categoryId)}" aria-current="${step.categoryId === activeCategory ? 'true' : 'false'}">
            <span class="docs-route__number">${index + 1}</span>
            <span class="docs-route__copy">
              <strong>${escapeHtml(title)}</strong>
              ${step.text ? `<small>${escapeHtml(step.text)}</small>` : ''}
            </span>
          </button>
        `;
      }).join('')}
    </div>
  `;

  map.querySelectorAll('[data-cat]').forEach((button) => {
    button.addEventListener('click', () => onSelect(button.dataset.cat));
  });

  return map;
}

export function createSectionBrief(meta, ui) {
  if (!meta || typeof meta !== 'object') return null;

  const hasSubtitle = Boolean(meta.subtitle);
  const rows = [
    meta.contains ? [ui.sectionContains || 'What is in this block', meta.contains] : null,
    meta.proves ? [ui.sectionProves || 'What it helps show', meta.proves] : null,
    meta.purpose ? [ui.sectionPurpose || 'Why it matters', meta.purpose] : null,
  ].filter(Boolean);

  if (!hasSubtitle && rows.length === 0) return null;

  const brief = document.createElement('section');
  brief.className = 'docs-brief';
  brief.innerHTML = `
    ${meta.subtitle ? `<p class="docs-brief__lead">${escapeHtml(meta.subtitle)}</p>` : ''}
    ${rows.length ? `
      <div class="docs-brief__grid">
        ${rows.map(([label, text]) => `
          <div class="docs-brief__item">
            <span class="docs-brief__label">${escapeHtml(label)}</span>
            <p>${escapeHtml(text)}</p>
          </div>
        `).join('')}
      </div>
    ` : ''}
  `;
  return brief;
}

export function getFilterOptions(ui) {
  return {
    types: [
      { value: 'all', label: ui.filterAll || 'All' },
      { value: 'request', label: ui.filterRequest || 'Request / appeal' },
      { value: 'response', label: ui.filterResponse || 'Response / refusal' },
      { value: 'decision', label: ui.filterDecision || 'Decision' },
      { value: 'evidence', label: ui.filterEvidence || 'Evidence' },
    ],
    institutions: [
      { value: 'all', label: ui.filterAll || 'All' },
      { value: 'court', label: ui.filterCourt || 'Court' },
      { value: 'prosecutor', label: ui.filterProsecutor || 'Prosecutor' },
      { value: 'mvd', label: ui.filterMvd || 'MVD / GSU' },
      { value: 'upch', label: ui.filterUpch || 'UPCH' },
      { value: 'fsb', label: ui.filterFsb || 'FSB' },
      { value: 'president-rf', label: ui.filterPresident || 'President RF' },
      { value: 'sovet-federatsii', label: ui.filterSenate || 'Senate' },
      { value: 'serbia', label: ui.filterSerbia || 'Serbia' },
      { value: 'europe', label: ui.filterEurope || 'Europe' },
      { value: 'unhcr', label: ui.filterUnhcr || 'UNHCR' },
      { value: 'echr', label: ui.filterEchr || 'ECHR' },
      { value: 'ukraine', label: ui.filterUkraine || 'Ukraine' },
      { value: 'vatican', label: ui.filterVatican || 'Vatican' },
      { value: 'political-support', label: ui.filterPoliticalSupport || 'Political support' },
      { value: 'asylum', label: ui.filterAsylum || 'Asylum' },
      { value: 'serbian-party', label: ui.filterParty || 'Serbian parties' },
      { value: 'interpol', label: ui.filterInterpol || 'Interpol' },
      { value: 'evidence', label: ui.filterEvidenceLabel || 'Evidence' },
      { value: 'complaints', label: ui.filterComplaints || 'Complaints' },
    ],
    variants: [
      { value: 'all', label: ui.filterAll || 'All' },
      { value: 'original', label: ui.filterOriginal || 'Original' },
      { value: 'translation', label: ui.filterTranslation || 'Translation' },
      { value: 'serbian', label: ui.filterSerbian || 'Serbian version' },
    ]
  };
}
