import { resolveDocMeta, resolveI18n } from '../../utils/resolve-i18n.js';

export function getFileType(filename) {
  if (!filename) return 'unknown';
  const ext = filename.split('.').pop().toLowerCase();
  if (ext === 'pdf') return 'pdf';
  if (ext === 'doc' || ext === 'docx') return 'word';
  return 'unknown';
}

export function fileTypeBadge(type, ui) {
  const map = { pdf: ui.badgePdf, word: ui.badgeWord };
  return map[type] || '';
}

export function buildFileUrl(docRowOrCategoryId, subcategoryId, filename) {
  if (typeof docRowOrCategoryId === 'object') {
    const parts = ['.', 'files', docRowOrCategoryId.fileBasePath || docRowOrCategoryId.categoryId];
    if (docRowOrCategoryId.fileSubpath || docRowOrCategoryId.subcategoryId) {
      parts.push(docRowOrCategoryId.fileSubpath || docRowOrCategoryId.subcategoryId);
    }
    parts.push(docRowOrCategoryId.filename);
    return parts.join('/');
  }

  return subcategoryId
    ? `./files/${docRowOrCategoryId}/${subcategoryId}/${filename}`
    : `./files/${docRowOrCategoryId}/${filename}`;
}

const variantMap = {
  ru: 'original',
  sr: 'serbian',
  'sr-ru': 'translation',
  'en': 'translation',
  'en-ru': 'translation',
  fr: 'translation',
};

function getDocVariant(language) {
  if (!language) return 'original';
  return variantMap[language] || 'translation';
}

function getDocType(categoryId) {
  if (categoryId === 'core-evidence') return 'evidence';
  if (categoryId === 'interpol') return 'request';
  return 'request';
}

function getTypeLabel(type, ui) {
  if (type === 'request') return ui.filterRequest || 'Request';
  if (type === 'response') return ui.filterResponse || 'Response';
  if (type === 'decision') return ui.filterDecision || 'Decision';
  if (type === 'evidence') return ui.filterEvidence || 'Evidence';
  return type;
}

function getStageLabel(stage, ui) {
  if (stage === 'outgoing') return ui.stageOutgoing || 'Outgoing';
  if (stage === 'incoming') return ui.stageIncoming || 'Incoming';
  if (stage === 'receipt') return ui.stageReceipt || 'Receipt';
  if (stage === 'court') return ui.stageCourt || 'Court step';
  if (stage === 'evidence') return ui.stageEvidence || 'Evidence';
  if (stage === 'translation') return ui.stageTranslation || 'Translation';
  if (stage === 'support') return ui.stageSupport || 'Support';
  if (stage === 'bundle') return ui.stageBundle || 'Bundle';
  if (stage === 'draft') return ui.stageDraft || 'Draft';
  return '';
}

function getDocSource(categoryId) {
  if (categoryId === 'interpol') return 'interpol';
  if (categoryId === 'core-evidence') return 'evidence';
  return categoryId;
}

function getSourceLabel(source, ui) {
  switch (source) {
    case 'court': return ui.filterCourt || 'Court';
    case 'prosecutor': return ui.filterProsecutor || 'Prosecutor';
    case 'mvd': return ui.filterMvd || 'MVD / GSU';
    case 'upch': return ui.filterUpch || 'UPCH';
    case 'fsb': return ui.filterFsb || 'FSB';
    case 'president-rf': return ui.filterPresident || 'President';
    case 'sovet-federatsii': return ui.filterSenate || 'Senate';
    case 'serbia': return ui.filterSerbia || 'Serbia';
    case 'europe': return ui.filterEurope || 'Europe';
    case 'interpol': return ui.filterInterpol || 'Interpol';
    case 'unhcr': return ui.filterUnhcr || 'UNHCR';
    case 'echr': return ui.filterEchr || 'ECHR';
    case 'ukraine': return ui.filterUkraine || 'Ukraine';
    case 'vatican': return ui.filterVatican || 'Vatican';
    case 'political-support': return ui.filterPoliticalSupport || 'Political support';
    case 'asylum': return ui.filterAsylum || 'Asylum';
    case 'party':
    case 'serbian-party': return ui.filterParty || 'Serbian parties';
    case 'evidence': return ui.filterEvidenceLabel || 'Evidence';
    case 'complaints': return ui.filterComplaints || 'Complaints';
    default: return source;
  }
}

export function flattenCatalog(catalog, ui) {
  const rows = [];
  for (const cat of catalog.categories) {
    const catLabel = resolveI18n(ui.__i18n, cat.title_i18n_key);
    if (cat.documents) {
      for (const doc of cat.documents) {
        const type = doc.type || getDocType(cat.id);
        const source = doc.source || getDocSource(cat.id);
        rows.push({
          categoryId: cat.id,
          subcategoryId: null,
          fileBasePath: cat.fileBasePath || cat.id,
          fileSubpath: cat.fileSubpath || null,
          subcategoryKey: null,
          type,
          typeLabel: getTypeLabel(type, ui),
          source,
          sourceLabel: getSourceLabel(source, ui),
          stage: doc.stage || '',
          stageLabel: getStageLabel(doc.stage, ui),
          trackLabel: typeof catLabel === 'string' ? catLabel : catLabel?.title || cat.id,
          variant: getDocVariant(doc.language),
          threadParentId: doc.threadParentId || null,
          ...doc,
        });
      }
    }
    if (cat.subcategories) {
      for (const sub of cat.subcategories) {
        const subLabel = resolveI18n(ui.__i18n, sub.title_i18n_key);
        for (const doc of sub.documents || []) {
          const type = doc.type || getDocType(cat.id);
          const source = doc.source || getDocSource(cat.id);
          rows.push({
            categoryId: cat.id,
            subcategoryId: sub.id,
            fileBasePath: cat.fileBasePath || cat.id,
            fileSubpath: sub.fileSubpath || sub.id,
            subcategoryKey: sub.title_i18n_key,
            type,
            typeLabel: getTypeLabel(type, ui),
            source,
            sourceLabel: getSourceLabel(source, ui),
            stage: doc.stage || '',
            stageLabel: getStageLabel(doc.stage, ui),
            trackLabel: typeof subLabel === 'string' ? subLabel : subLabel?.title || sub.id,
            variant: getDocVariant(doc.language),
            threadParentId: doc.threadParentId || null,
            ...doc,
          });
        }
      }
    }
  }
  return rows;
}

export function groupRelatedDocs(docs, i18n) {
  const byGroup = new Map();
  const ungrouped = [];

  docs.forEach((doc) => {
    const meta = resolveDocMeta(i18n, doc.title_i18n_key);
    if (!meta.group) {
      ungrouped.push(doc);
      return;
    }
    if (!byGroup.has(meta.group)) byGroup.set(meta.group, []);
    byGroup.get(meta.group).push(doc);
  });

  return { byGroup, ungrouped };
}
