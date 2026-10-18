// FB2 reader: a continuous, reflowable text page. "Pages" are virtual (text length / 1800) so progress survives font and screen changes.
import { loadBook } from './load.js';
import { createRenderer } from './render.js';
import { getPref, setPref } from '../../core/prefs.js';
const MIN = 0.7, MAX = 2.6;
export async function mountFb2({ container, scroller, blob, startPage = 1, onPage }) {
  const book = await loadBook(blob);
  const r = createRenderer(book);
  const root = r.render();
  let scale = getPref('fb2scale', 1), raf = 0, jumpBack = null;
  root.style.setProperty('--fs', scale);
  container.replaceChildren(root);
  const range = () => Math.max(1, scroller.scrollHeight - scroller.clientHeight);
  const goTo = (n) => { scroller.scrollTop = ((Math.min(Math.max(n, 1), book.pages) - 1) / book.pages) * range(); };
  const here = () => Math.min(book.pages, 1 + Math.floor((scroller.scrollTop / range()) * book.pages));
  const onScroll = () => { raf ||= requestAnimationFrame(() => { raf = 0; onPage(here(), book.pages); }); };
  const back = Object.assign(document.createElement('button'), { className: 'fb2-back', hidden: true, textContent: '↩' });
  document.getElementById('viewer').append(back);
  back.addEventListener('click', () => { scroller.scrollTop = jumpBack ?? scroller.scrollTop; back.hidden = true; });
  const onClick = (e) => {
    const to = e.target.closest('a[data-to]')?.dataset.to;
    const target = to && root.querySelector(`[id="${CSS.escape(to)}"]`);
    if (!target) return;
    jumpBack = scroller.scrollTop;
    target.scrollIntoView({ block: 'start' });
    back.hidden = false;
  };
  root.addEventListener('click', onClick);
  scroller.addEventListener('scroll', onScroll, { passive: true });
  requestAnimationFrame(() => { goTo(startPage); onPage(here(), book.pages); });
  return {
    caps: { thumbs: false, draw: false, mode: false },
    zoomBy(f) {
      const frac = scroller.scrollTop / range();
      scale = Math.min(MAX, Math.max(MIN, scale * f));
      root.style.setProperty('--fs', scale);
      setPref('fb2scale', scale);
      scroller.scrollTop = frac * range();
    },
    goTo,
    destroy() { scroller.removeEventListener('scroll', onScroll); cancelAnimationFrame(raf); back.remove(); r.dispose(); container.replaceChildren(); },
  };
}
