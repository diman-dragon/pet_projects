/**
 * highlight.js — utilities for highlighting search terms in rendered DOM.
 */

import { escapeHtml } from './security.js';

/**
 * Walk all text nodes inside `element` and wrap occurrences of `term`
 * in <mark> elements with the accent colour.
 */
export function highlightTextInElement(element, term) {
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null);
  const nodes = [];
  let node;
  while (node = walker.nextNode()) nodes.push(node);

  nodes.forEach(node => {
    const parent = node.parentNode;
    if (parent.nodeName === 'MARK') return;
    const text = node.textContent;
    if (!text.toLowerCase().includes(term.toLowerCase())) return;

    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const parts = text.split(new RegExp(`(${escaped})`, 'gi'));
    const fragment = document.createDocumentFragment();
    parts.forEach(part => {
      if (part.toLowerCase() === term.toLowerCase()) {
        const mark = document.createElement('mark');
        mark.textContent = part;
        mark.style.backgroundColor = 'var(--accent)';
        mark.style.color = 'var(--accent-soft)';
        fragment.appendChild(mark);
      } else {
        fragment.appendChild(document.createTextNode(part));
      }
    });
    parent.replaceChild(fragment, node);
  });
}

/**
 * Remove all <mark> wrappers previously inserted by highlightTextInElement.
 */
export function clearHighlights(element) {
  element.querySelectorAll('mark').forEach(mark => {
    const parent = mark.parentNode;
    if (!parent) return;
    parent.replaceChild(document.createTextNode(mark.textContent), mark);
    parent.normalize();
  });
}

/**
 * Highlight `term` inside an already HTML-escaped string.
 * The input MUST be escaped before calling this function.
 */
export function highlightSnippet(escapedText, term) {
  const escapedTermHtml = escapeHtml(term).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return escapedText.replace(
    new RegExp(`(${escapedTermHtml})`, 'gi'),
    '<mark style="background:var(--accent);color:var(--accent-soft)">$1</mark>'
  );
}
