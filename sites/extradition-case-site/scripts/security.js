/**
 * security.js — HTML escaping and URL validation utilities.
 * Import these helpers in any module that places external data into innerHTML.
 */

/**
 * Escape a string for safe insertion as HTML text content.
 * Call this on every untrusted value before using it in innerHTML.
 */
export function escapeHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Validate that a URL is http/https or a relative path.
 * Returns the URL if safe, '#' otherwise.
 * Prevents javascript:, data:, and other dangerous schemes.
 */
export function safeUrl(url) {
  if (!url || typeof url !== 'string') return '#';
  const trimmed = url.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  if (/^\.?\//.test(trimmed)) return trimmed;
  if (/^#/.test(trimmed)) return trimmed;
  return '#';
}

/**
 * Sanitize a tiny subset of HTML.
 * Preserves safe links plus basic emphasis and line breaks.
 * Any other element is flattened to text content.
 */
export function sanitizeRichText(html) {
  if (!html) return '';
  const template = document.createElement('template');
  template.innerHTML = html;

  const elements = [...template.content.querySelectorAll('*')];
  elements.forEach((el) => {
    if (el.tagName === 'BR') return;

    if (el.tagName === 'STRONG' || el.tagName === 'EM') {
      const safeEl = document.createElement(el.tagName.toLowerCase());
      safeEl.textContent = el.textContent || '';
      el.replaceWith(safeEl);
      return;
    }

    if (el.tagName !== 'A') {
      el.replaceWith(document.createTextNode(el.textContent || ''));
      return;
    }

    const href = safeUrl(el.getAttribute('href') || '');
    if (href === '#') {
      el.replaceWith(document.createTextNode(el.textContent || ''));
      return;
    }

    const text = el.textContent || '';
    const link = document.createElement('a');
    link.href = href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = text;
    el.replaceWith(link);
  });

  return template.innerHTML;
}

/**
 * Validate an email address (basic pattern check).
 * Prevents javascript: injection via mailto: hrefs.
 */
export function safeEmail(email) {
  if (!email || typeof email !== 'string') return '';
  const t = email.trim();
  return /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(t) ? t : '';
}

/**
 * Validate a Telegram or http/https URL.
 */
export function safeTelegramUrl(url) {
  if (!url || typeof url !== 'string') return '#';
  const t = url.trim();
  if (/^https?:\/\//i.test(t) || /^tg:/i.test(t)) return t;
  return '#';
}
