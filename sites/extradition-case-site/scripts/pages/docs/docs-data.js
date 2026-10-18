import { resolveDocMeta } from '../../utils/resolve-i18n.js';

export function normalizeDate(date) {
  if (!date) return '0000-01-01';
  return date.length === 4 ? `${date}-01-01` : date;
}

export function sortDocs(docs, i18n) {
  const variantOrder = { original: 0, translation: 1, serbian: 2 };
  return [...docs].sort((a, b) => {
    const metaA = resolveDocMeta(i18n, a.title_i18n_key);
    const metaB = resolveDocMeta(i18n, b.title_i18n_key);
    const dateDiff = normalizeDate(metaB.date).localeCompare(normalizeDate(metaA.date));
    if (dateDiff !== 0) return dateDiff;
    return (variantOrder[a.variant || 'original'] || 0) - (variantOrder[b.variant || 'original'] || 0);
  });
}

export function filterDocuments(docs, filters) {
  const { activeType, activeInstitution, activeVariant } = filters;
  return docs.filter((doc) => {
    if (activeType !== 'all' && doc.type !== activeType) return false;
    if (activeInstitution !== 'all' && doc.source !== activeInstitution) return false;
    if (activeVariant !== 'all' && doc.variant !== activeVariant) return false;
    return true;
  });
}
