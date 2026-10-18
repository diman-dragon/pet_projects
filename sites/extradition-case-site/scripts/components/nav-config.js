export const PAGE_IDS = ['home', 'timeline', 'legal', 'persons', 'docs', 'intl', 'media', 'flagrant', 'article8'];

export const NAV_GROUPS = [
  { id: 'overview', pages: ['home', 'timeline'] },
  { id: 'analysis', pages: ['legal', 'intl', 'flagrant', 'article8', 'persons'] },
  { id: 'archive', pages: ['docs', 'media'] },
];

export const NAV_GROUP_LABELS = {
  ru: {
    overview: 'Обзор',
    analysis: 'Правовая рамка',
    archive: 'Архив и медиа',
  },
  en: {
    overview: 'Overview',
    analysis: 'Legal frame',
    archive: 'Archive and media',
  },
  sr: {
    overview: 'Pregled',
    analysis: 'Pravni okvir',
    archive: 'Arhiva i mediji',
  },
};

export function getGroupLabels(lang) {
  return NAV_GROUP_LABELS[lang] || NAV_GROUP_LABELS.en;
}
