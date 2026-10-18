// FictionBook XML -> readable HTML (built with DOM calls only, so book text can never inject markup).
import { hrefOf } from './load.js';
const INLINE = { strong: 'strong', emphasis: 'em', strikethrough: 's', sub: 'sub', sup: 'sup', code: 'code', style: 'span', a: 'a' };
const BLOCK = { p: ['p', ''], subtitle: ['p', 'subtitle'], 'text-author': ['p', 'text-author'], v: ['p', 'v'], epigraph: ['blockquote', 'epigraph'], cite: ['blockquote', 'cite'], poem: ['div', 'poem'], stanza: ['div', 'stanza'], annotation: ['div', 'annotation'], table: ['table', ''], tr: ['tr', ''], th: ['th', ''], td: ['td', ''] };
const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };
export function createRenderer(book) {
  const urls = new Map();
  function imageUrl(id) {
    if (urls.has(id)) return urls.get(id);
    const b = book.binaries.get(id);
    if (!b) return '';
    const bin = atob(b.b64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const url = URL.createObjectURL(new Blob([bytes], { type: b.type }));
    urls.set(id, url);
    return url;
  }
  function node(n, depth) {
    if (n.nodeType === 3) return document.createTextNode(n.nodeValue);
    if (n.nodeType !== 1) return null;
    const name = n.localName;
    if (name === 'image') { const src = imageUrl(hrefOf(n)); if (!src) return null; const i = el('img', 'pic'); i.src = src; i.loading = 'lazy'; i.alt = n.getAttribute('alt') ?? ''; return i; }
    if (name === 'empty-line') return el('div', 'gap');
    if (name === 'title') return heading(n, depth);
    if (name === 'section') return section(n, depth + 1);
    const spec = BLOCK[name] ?? (INLINE[name] ? [INLINE[name], name === 'a' && n.getAttribute('type') === 'note' ? 'note' : ''] : null);
    if (!spec) return kids(el('div'), n, depth);
    const out = el(spec[0], spec[1]);
    if (name === 'a') { const id = hrefOf(n); if (id && !/^[a-z]+:/i.test(id)) out.dataset.to = id; }
    return kids(out, n, depth);
  }
  function kids(out, n, depth) {
    for (const c of n.childNodes) { const r = node(c, depth); if (r) out.append(r); }
    return out;
  }
  function heading(n, depth) {
    const h = el(`h${Math.min(6, depth + 1)}`, 'title');
    h.append(...[...n.children].map((p) => node(p, depth)).filter(Boolean).flatMap((x, i) => (i ? [document.createElement('br'), x] : [x])));
    return h;
  }
  function section(n, depth) {
    const s = el('section', 'sec');
    if (n.id || n.getAttribute('id')) s.id = n.getAttribute('id');
    return kids(s, n, depth);
  }
  function body(b) {
    const out = el('div', b.getAttribute('name') === 'notes' ? 'notes' : 'body');
    if (b.getAttribute('name') === 'notes') { const t = el('h2', 'title'); t.textContent = '* * *'; out.append(t); }
    return kids(out, b, 1);
  }
  function render() {
    const root = el('article', 'fb2');
    root.lang = book.lang;
    const head = el('header', 'fb2-head');
    if (book.authors.length) head.append(Object.assign(el('div', 'fb2-author'), { textContent: book.authors.join(', ') }));
    if (book.title) head.append(Object.assign(el('h1', 'fb2-title'), { textContent: book.title }));
    if (book.coverId && imageUrl(book.coverId)) { const c = el('img', 'pic fb2-cover'); c.src = imageUrl(book.coverId); head.append(c); }
    root.append(head);
    if (book.annotation) root.append(kids(el('div', 'annotation'), book.annotation, 1));
    book.bodies.forEach((b) => root.append(body(b)));
    root.querySelectorAll('[id]').forEach(() => {});
    return root;
  }
  return { render, dispose: () => urls.forEach((u) => URL.revokeObjectURL(u)) };
}
