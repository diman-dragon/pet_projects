/** Resolve dot-path keys against nested i18n objects. */
export function resolveI18n(root, keyPath) {
  if (!root || !keyPath) return undefined;
  return keyPath.split('.').reduce((obj, key) => (obj != null ? obj[key] : undefined), root);
}

/** Leaf may be a string (title only) or { title, desc, date, highlight, highlight_label }. */
export function resolveDocMeta(i18n, keyPath) {
  const leaf = resolveI18n(i18n, keyPath);
  if (!leaf) return { title: keyPath, desc: '', date: '' };
  if (typeof leaf === 'string') return { title: leaf, desc: '', date: '' };
  return {
    title: leaf.title || '',
    desc: leaf.desc || '',
    date: leaf.date || '',
    highlight: !!leaf.highlight,
    highlight_label: leaf.highlight_label || '',
    group: leaf.group || '',
  };
}
