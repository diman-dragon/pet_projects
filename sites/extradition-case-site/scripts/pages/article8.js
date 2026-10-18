import { store } from '../store.js';
import { escapeHtml, sanitizeRichText } from '../security.js';
import { createPageShell, appendPageSummary } from '../components/page-shell.js';
import { getPageAside } from '../components/page-asides.js';
import { createRecordRow, createSectionHeading, createStatsGrid, documentCardHtml, splitRichText } from '../components/record-layout.js';

function renderFamilyFact(fact) {
  return `
    <article class="record-episode">
      <div class="record-episode__label">${escapeHtml(fact.label)}</div>
      <div class="record-episode__body">
        <p>${escapeHtml(fact.text)}</p>
      </div>
    </article>
  `;
}

function renderEchrPara(para) {
  const text = typeof para === 'string' ? para : para.text || '';
  const isHtml = typeof para === 'object' && para.is_html;
  return `<p>${isHtml ? sanitizeRichText(text) : escapeHtml(text)}</p>`;
}

function renderFailure(failure, labels) {
  return `
    <article class="record-episode">
      <div class="record-episode__label">${escapeHtml(failure.tag)}</div>
      <div class="record-episode__body">
        <div class="record-focus">
          <span class="record-focus__label">${escapeHtml(labels.what_happened)}</span>
          ${splitRichText(failure.what_happened)}
        </div>
        <div class="record-focus record-focus--warning">
          <span class="record-focus__label">${escapeHtml(labels.what_missing)}</span>
          ${splitRichText(failure.what_missing)}
        </div>
        <div class="record-focus">
          <span class="record-focus__label">${escapeHtml(labels.why_matters)}</span>
          ${splitRichText(failure.why_matters)}
        </div>
      </div>
    </article>
  `;
}

function renderProportionalityItem(item, labels) {
  return `
    <article class="record-episode">
      <div class="record-episode__label">${escapeHtml(item.criterion)}</div>
      <div class="record-episode__body">
        <div class="record-focus">
          <span class="record-focus__label">${escapeHtml(labels.assessment)}</span>
          ${splitRichText(item.assessment)}
        </div>
        ${item.doc ? documentCardHtml(item.doc, item.doc_label || 'Document') : ''}
      </div>
    </article>
  `;
}

export async function renderArticle8Page(container) {
  const lang = store.state.lang;
  let response = await fetch(`./scripts/data/i18n/article8/${lang}.json`);
  if (!response.ok) response = await fetch('./scripts/data/i18n/article8/ru.json');
  const t = await response.json();
  const aside = getPageAside('article8', lang);

  const { body, after } = createPageShell(container, {
    pageClass: 'flagrant-page',
    badge: t.badge,
    title: t.title,
    subtitle: t.subtitle,
    intro: t.intro,
    asideLabel: aside.label,
    asideText: aside.text,
  });


  body.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Фактическая основа' : lang === 'sr' ? 'Činjenična osnova' : 'Factual basis',
    title: t.family_block.title,
  }));

  const familySection = document.createElement('section');
  const familyIntro = document.createElement('p');
  familyIntro.className = 'record-intro';
  familyIntro.textContent = t.family_block.intro;
  familySection.appendChild(familyIntro);

  const factsEl = document.createElement('div');
  factsEl.className = 'record-episodes';
  factsEl.innerHTML = t.family_block.facts.map(renderFamilyFact).join('');
  familySection.appendChild(factsEl);
  body.appendChild(familySection);

  body.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Правовой стандарт' : lang === 'sr' ? 'Pravni standard' : 'Legal standard',
    title: t.echr_block.title,
  }));

  body.appendChild(createRecordRow({
    eyebrow: 'ECtHR',
    status: lang === 'ru' ? 'Статья 8 ЕКПЧ' : lang === 'sr' ? 'Član 8 EKLJP' : 'Article 8 ECHR',
    tone: 'danger',
    bodyHtml: t.echr_block.paras.map(renderEchrPara).join(''),
  }));

  body.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Системный пробел' : lang === 'sr' ? 'Sistemski propust' : 'Systemic gap',
    title: t.failure_title,
  }));

  const failureDesc = document.createElement('p');
  failureDesc.className = 'record-intro';
  failureDesc.textContent = t.failure_desc;
  body.appendChild(failureDesc);

  t.failures.forEach((failure) => {
    body.appendChild(createRecordRow({
      eyebrow: lang === 'ru' ? 'Инстанция' : lang === 'sr' ? 'Organ' : 'Authority',
      title: failure.title,
      tone: 'danger',
      bodyHtml: renderFailure(failure, t.labels),
    }));
  });

  body.appendChild(createSectionHeading({
    kicker: lang === 'ru' ? 'Тест соразмерности' : lang === 'sr' ? 'Test proporcionalnosti' : 'Proportionality test',
    title: t.proportionality_block.title,
  }));

  const propIntro = document.createElement('p');
  propIntro.className = 'record-intro';
  propIntro.textContent = t.proportionality_block.intro;
  body.appendChild(propIntro);

  const propSection = document.createElement('div');
  propSection.className = 'record-episodes';
  propSection.innerHTML = t.proportionality_block.items
    .map((item) => renderProportionalityItem(item, t.labels))
    .join('');
  body.appendChild(propSection);

  body.appendChild(createRecordRow({
    eyebrow: lang === 'ru' ? 'Ответственность' : lang === 'sr' ? 'Odgovornost' : 'Responsibility',
    status: t.serbia_obligation.title,
    tone: 'info',
    bodyHtml: `<p>${escapeHtml(t.serbia_obligation.text)}</p>`,
  }));

  body.appendChild(createRecordRow({
    eyebrow: lang === 'ru' ? 'Итог' : lang === 'sr' ? 'Zaključak' : 'Conclusion',
    status: t.conclusion_title,
    tone: 'danger',
    bodyHtml: t.conclusion_paras.map((p) => `<p>${escapeHtml(p)}</p>`).join(''),
  }));

  appendPageSummary(after, t.summary, 'danger');
}
