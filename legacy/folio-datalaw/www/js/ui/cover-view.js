// Lazy covers: requests a thumbnail when a card nears the viewport, cancels the request when it leaves.
import { peekCover, requestCover, cancelCover } from '../app/cover-loader.js';
const metas = new WeakMap();
let io = null;
function show(li, url) {
  const img = li.querySelector('img');
  if (!url || !img) return;
  img.src = url;
  img.hidden = false;
  li.querySelector('.cover-ph').hidden = true;
}
export function watchCovers(root, cards) {
  io?.disconnect();
  io = new IntersectionObserver((entries) => entries.forEach((x) => {
    const meta = metas.get(x.target);
    if (x.isIntersecting) requestCover(meta).then((url) => show(x.target, url)); else cancelCover(meta.id);
  }), { root, rootMargin: '300px 0px' });
  cards.forEach(({ li, meta }) => {
    metas.set(li, meta);
    const url = peekCover(meta);
    if (url) show(li, url); else io.observe(li);
  });
}
