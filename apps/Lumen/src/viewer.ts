import { fullUrl } from './native';
import { store } from './store';
import { thumbs } from './thumbs';
import { fmtDateTime, t } from './i18n';
import { h, icon, iconBtn, menu, pushBack } from './ui';
import { hideFolder, moveItems, openWith, renameItem, shareItems, showInfo, toggleFavorite, trashItems } from './actions';
import type { Item } from './types';

interface Pane {
  el: HTMLElement;
  key: string;
  media: HTMLImageElement | HTMLVideoElement | null;
}

const MAX_SCALE = 6;
const TAP_MS = 300;

/**
 * Полноэкранный просмотрщик: свайп между файлами, pinch/double-tap зум,
 * закрытие свайпом вниз, видео. Жесты — на pointer events, без зависимостей.
 */
export function openViewer(list: Item[], start: number, onClose?: (last: Item | undefined) => void): void {
  let L = list.slice();
  let idx = Math.max(0, Math.min(start, L.length - 1));
  let dx = 0; // смещение ленты при свайпе
  let scale = 1;
  let tx = 0;
  let ty = 0;
  let dismissY = 0;
  let animating = false;
  let closed = false;

  const root = h('div', { class: 'viewer' });
  const track = h('div', { class: 'vtrack' });
  let P: Pane[] = [0, 1, 2].map(() => ({ el: h('div', { class: 'vpane' }), key: '', media: null }));
  track.append(...P.map((p) => p.el));

  const title = h('div', { class: 'vt-name' });
  const sub = h('div', { class: 'vt-sub' });
  const favBtn = iconBtn('star', t('favorite'), () => toggleFavorite([cur()]));
  const top = h(
    'div',
    { class: 'vtop' },
    iconBtn('back', t('back'), () => close()),
    h('div', { class: 'vt-text' }, title, sub),
    iconBtn('more', t('more'), () => openMore())
  );
  const bottom = h(
    'div',
    { class: 'vbottom' },
    iconBtn('share', t('share'), () => void shareItems([cur()])),
    favBtn,
    iconBtn('info', t('info'), () => void showInfo(cur())),
    iconBtn('trash', t('delete'), () => void trashItems([cur()]))
  );
  root.append(track, top, bottom);
  document.getElementById('overlays')!.append(root);
  requestAnimationFrame(() => root.classList.add('on'));

  const W = (): number => root.clientWidth || window.innerWidth;
  const H = (): number => root.clientHeight || window.innerHeight;
  const cur = (): Item => {
    const it = L[idx];
    return store.byKey.get(it.k) ?? it;
  };

  // ------------------------------------------------------------ рендер

  function fill(p: Pane, it: Item | undefined): void {
    p.el.textContent = '';
    p.media = null;
    p.key = it?.k ?? '';
    if (!it) return;
    if (it.v) {
      const v = h('video', { class: 'vmedia', controls: true, playsInline: true, preload: 'metadata' });
      const tu = thumbs.url(it.k);
      if (tu) v.poster = tu;
      p.media = v;
      p.el.append(v);
    } else {
      const img = h('img', { class: 'vmedia', draggable: false, decoding: 'async', alt: it.name });
      const tu = thumbs.url(it.k);
      if (tu) img.src = tu;
      p.media = img;
      p.el.append(img);
      const full = new Image();
      full.decoding = 'async';
      const key = it.k;
      full.onload = () => {
        if (p.key === key) img.src = full.src;
      };
      full.src = fullUrl(it);
    }
  }

  function activate(): void {
    P.forEach((p, i) => {
      const m = p.media;
      if (!(m instanceof HTMLVideoElement)) return;
      if (i === 1) {
        if (!m.getAttribute('src')) {
          const it = L[idx];
          if (it) m.src = fullUrl(it);
        }
      } else if (!m.paused) m.pause();
    });
  }

  function positionPanes(): void {
    const w = W();
    P.forEach((p, i) => {
      p.el.style.transform = `translate3d(${(i - 1) * w + dx}px,0,0)`;
    });
  }

  function applyZoom(): void {
    const m = P[1].media;
    if (m) m.style.transform = `translate3d(${tx}px,${ty}px,0) scale(${scale})`;
    root.classList.toggle('zoomed', scale > 1.02);
  }

  function resetZoom(): void {
    scale = 1;
    tx = ty = 0;
    P.forEach((p) => {
      if (p.media) p.media.style.transform = '';
    });
    root.classList.remove('zoomed');
  }

  function updateBars(): void {
    const it = cur();
    title.textContent = it.name;
    sub.textContent = `${idx + 1} / ${L.length} · ${fmtDateTime(it.taken)}`;
    const isFav = store.fav.has(it.k);
    favBtn.replaceChildren(icon(isFav ? 'starFill' : 'star'));
    favBtn.classList.toggle('on', isFav);
  }

  function render(): void {
    for (let o = -1; o <= 1; o++) fill(P[o + 1], L[idx + o]);
    dx = 0;
    resetZoom();
    positionPanes();
    activate();
    updateBars();
  }

  // ------------------------------------------------------------ навигация

  function animateTrack(on: boolean): void {
    track.classList.toggle('anim', on);
  }

  function go(delta: 1 | -1): void {
    const ni = idx + delta;
    if (ni < 0 || ni >= L.length) {
      snapBack();
      return;
    }
    animating = true;
    animateTrack(true);
    dx = -delta * W();
    positionPanes();
    setTimeout(() => {
      animateTrack(false);
      idx = ni;
      if (delta === 1) P.push(P.shift()!);
      else P.unshift(P.pop()!);
      fill(P[delta === 1 ? 2 : 0], L[idx + delta]);
      dx = 0;
      resetZoom();
      positionPanes();
      activate();
      updateBars();
      animating = false;
    }, 230);
  }

  function snapBack(): void {
    animateTrack(true);
    dx = 0;
    positionPanes();
    setTimeout(() => animateTrack(false), 230);
  }

  function close(): void {
    if (closed) return;
    closed = true;
    unback();
    offStore();
    offFav();
    document.removeEventListener('keydown', onKey);
    P.forEach((p) => {
      if (p.media instanceof HTMLVideoElement) p.media.pause();
    });
    root.classList.remove('on');
    root.classList.add('closing');
    const last = L[idx];
    setTimeout(() => root.remove(), 200);
    onClose?.(last);
  }
  const unback = pushBack(close);

  function openMore(): void {
    const it = cur();
    menu(it.name, [
      { icon: 'edit', label: t('rename'), onClick: () => void renameItem(it) },
      { icon: 'move', label: t('move'), onClick: () => void moveItems([it], false) },
      { icon: 'copy', label: t('copy'), onClick: () => void moveItems([it], true) },
      { icon: 'open', label: t('openWith'), onClick: () => void openWith(it) },
      it.dir ? { icon: 'eyeOff', label: t('hideFolder'), onClick: () => hideFolder(it.dir) } : null,
    ]);
  }

  // ------------------------------------------------------------ масштаб

  function fitSize(): [number, number] {
    const it = L[idx];
    const m = P[1].media;
    let iw = it.w;
    let ih = it.h;
    if (m instanceof HTMLImageElement && m.naturalWidth) {
      iw = m.naturalWidth;
      ih = m.naturalHeight;
    } else if (m instanceof HTMLVideoElement && m.videoWidth) {
      iw = m.videoWidth;
      ih = m.videoHeight;
    }
    if (!iw || !ih) return [W(), H()];
    const f = Math.min(W() / iw, H() / ih);
    return [iw * f, ih * f];
  }

  function bounds(s: number): [number, number] {
    const [fw, fh] = fitSize();
    return [Math.max(0, (fw * s - W()) / 2), Math.max(0, (fh * s - H()) / 2)];
  }

  const clamp = (v: number, a: number, b: number): number => Math.min(b, Math.max(a, v));

  function animateZoom(s: number, x: number, y: number): void {
    const m = P[1].media;
    if (!m) return;
    m.style.transition = 'transform .24s cubic-bezier(.2,.8,.2,1)';
    scale = s;
    const [bx, by] = bounds(s);
    tx = clamp(x, -bx, bx);
    ty = clamp(y, -by, by);
    applyZoom();
    setTimeout(() => {
      m.style.transition = '';
    }, 260);
  }

  function toggleChrome(): void {
    root.classList.toggle('hide-chrome');
  }

  // ------------------------------------------------------------ жесты

  const ptrs = new Map<number, { x: number; y: number }>();
  type Mode = '' | 'swipe' | 'dismiss' | 'pan' | 'pinch' | 'ignore';
  let mode: Mode = '';
  let sx = 0;
  let sy = 0;
  let st = 0;
  let stx = 0;
  let sty = 0;
  let moved = false;
  let pd0 = 1;
  let ps0 = 1;
  let pm0 = { x: 0, y: 0 };
  let ptx0 = 0;
  let pty0 = 0;
  let lastX = 0;
  let lastY = 0;
  let lastT = 0;
  let vx = 0;
  let vy = 0;
  let lastTap = 0;
  let lastTapX = 0;
  let lastTapY = 0;
  let tapTimer = 0;

  const isVideo = (): boolean => P[1].media instanceof HTMLVideoElement;

  track.addEventListener('pointerdown', (e) => {
    if (animating) return;
    // нижняя полоса видео — нативные контролы
    if (e.target instanceof HTMLVideoElement && e.clientY > H() - 96) return;
    track.setPointerCapture(e.pointerId);
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (ptrs.size === 2 && !isVideo()) {
      const [a, b] = [...ptrs.values()];
      pd0 = Math.hypot(a.x - b.x, a.y - b.y) || 1;
      ps0 = scale;
      pm0 = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      ptx0 = tx;
      pty0 = ty;
      mode = 'pinch';
      dx = 0;
      positionPanes();
      return;
    }
    if (ptrs.size > 2) return;
    sx = lastX = e.clientX;
    sy = lastY = e.clientY;
    st = lastT = performance.now();
    stx = tx;
    sty = ty;
    moved = false;
    vx = vy = 0;
    mode = '';
    dismissY = 0;
  });

  track.addEventListener('pointermove', (e) => {
    if (!ptrs.has(e.pointerId)) return;
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (mode === 'pinch' && ptrs.size >= 2) {
      const [a, b] = [...ptrs.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const ns = clamp(ps0 * (dist / pd0), 0.85, MAX_SCALE);
      const cx = W() / 2;
      const cy = H() / 2;
      const k = ns / ps0;
      tx = mid.x - cx - (pm0.x - cx - ptx0) * k;
      ty = mid.y - cy - (pm0.y - cy - pty0) * k;
      scale = ns;
      applyZoom();
      return;
    }
    if (ptrs.size !== 1) return;

    const mx = e.clientX - sx;
    const my = e.clientY - sy;
    const now = performance.now();
    const dt = Math.max(1, now - lastT);
    vx = 0.8 * vx + 0.2 * ((e.clientX - lastX) / dt);
    vy = 0.8 * vy + 0.2 * ((e.clientY - lastY) / dt);
    lastX = e.clientX;
    lastY = e.clientY;
    lastT = now;

    if (mode === '') {
      if (Math.hypot(mx, my) < 10) return;
      moved = true;
      if (scale > 1.02) mode = 'pan';
      else if (Math.abs(mx) > Math.abs(my)) mode = 'swipe';
      else if (my > 0) mode = 'dismiss';
      else mode = 'ignore';
      if (mode !== 'ignore') {
        window.clearTimeout(tapTimer);
        lastTap = 0;
      }
    }

    if (mode === 'swipe') {
      const hasNeighbor = mx < 0 ? idx < L.length - 1 : idx > 0;
      dx = hasNeighbor ? mx : mx * 0.25;
      positionPanes();
    } else if (mode === 'dismiss') {
      dismissY = Math.max(0, my);
      const k = 1 - Math.min(0.25, dismissY / 1600);
      track.style.transform = `translate3d(${mx * 0.35}px,${dismissY}px,0) scale(${k})`;
      root.style.setProperty('--bg-a', String(1 - Math.min(0.85, dismissY / 420)));
      root.classList.add('dragging');
    } else if (mode === 'pan') {
      const [bx, by] = bounds(scale);
      const nx = stx + mx;
      const cx = clamp(nx, -bx, bx);
      const over = nx - cx;
      tx = cx;
      ty = clamp(sty + my, -by, by);
      const hasNeighbor = over < 0 ? idx < L.length - 1 : idx > 0;
      dx = over !== 0 ? (hasNeighbor ? over : over * 0.25) : 0;
      applyZoom();
      positionPanes();
    }
  });

  function endPointer(e: PointerEvent): void {
    if (!ptrs.has(e.pointerId)) return;
    ptrs.delete(e.pointerId);

    if (mode === 'pinch') {
      if (ptrs.size === 1) {
        // продолжаем как панорамирование оставшимся пальцем
        const [p] = [...ptrs.values()];
        sx = lastX = p.x;
        sy = lastY = p.y;
        stx = tx;
        sty = ty;
        mode = 'pan';
      } else if (ptrs.size === 0) {
        mode = '';
      }
      if (scale < 1.02) animateZoom(1, 0, 0);
      else {
        const [bx, by] = bounds(scale);
        animateZoom(scale, clamp(tx, -bx, bx), clamp(ty, -by, by));
      }
      return;
    }
    if (ptrs.size > 0) return;

    const dt = performance.now() - st;
    const w = W();
    if (mode === '' && !moved && dt < TAP_MS) {
      handleTap(e.clientX, e.clientY);
    } else if (mode === 'swipe' || (mode === 'pan' && dx !== 0)) {
      if ((dx < -w * 0.22 || vx < -0.55) && idx < L.length - 1) go(1);
      else if ((dx > w * 0.22 || vx > 0.55) && idx > 0) go(-1);
      else snapBack();
    } else if (mode === 'dismiss') {
      root.classList.remove('dragging');
      if (dismissY > 110 || vy > 0.7) {
        track.style.transition = 'transform .2s ease-in';
        track.style.transform = `translate3d(0,${H()}px,0) scale(.8)`;
        close();
      } else {
        track.style.transition = 'transform .2s ease-out';
        track.style.transform = '';
        root.style.setProperty('--bg-a', '1');
        setTimeout(() => {
          track.style.transition = '';
        }, 220);
      }
    }
    mode = '';
  }
  track.addEventListener('pointerup', endPointer);
  track.addEventListener('pointercancel', endPointer);

  function handleTap(x: number, y: number): void {
    const now = Date.now();
    if (now - lastTap < TAP_MS && Math.hypot(x - lastTapX, y - lastTapY) < 48 && !isVideo()) {
      window.clearTimeout(tapTimer);
      lastTap = 0;
      if (scale > 1.02) animateZoom(1, 0, 0);
      else {
        const s = 2.6;
        animateZoom(s, -(x - W() / 2) * (s - 1), -(y - H() / 2) * (s - 1));
      }
      return;
    }
    lastTap = now;
    lastTapX = x;
    lastTapY = y;
    window.clearTimeout(tapTimer);
    tapTimer = window.setTimeout(toggleChrome, isVideo() ? 0 : TAP_MS);
  }

  // колесо мыши — для отладки в браузере
  track.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      if (isVideo()) return;
      const s = clamp(scale * (e.deltaY < 0 ? 1.2 : 1 / 1.2), 1, MAX_SCALE);
      const k = s / scale;
      const [bx, by] = bounds(s);
      scale = s;
      tx = clamp(e.clientX - W() / 2 - (e.clientX - W() / 2 - tx) * k, -bx, bx);
      ty = clamp(e.clientY - H() / 2 - (e.clientY - H() / 2 - ty) * k, -by, by);
      applyZoom();
    },
    { passive: false }
  );

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'ArrowRight') go(1);
    else if (e.key === 'ArrowLeft') go(-1);
    else if (e.key === 'Escape') close();
  }
  document.addEventListener('keydown', onKey);

  // ------------------------------------------------------------ синхронизация с библиотекой

  const offStore = store.on('change', () => {
    if (closed) return;
    const current = L[idx];
    const next = L.filter((i) => store.byKey.has(i.k));
    if (next.length === L.length) {
      updateBars();
      return;
    }
    if (!next.length) {
      close();
      return;
    }
    L = next;
    const ni = current ? L.findIndex((i) => i.k === current.k) : -1;
    idx = ni >= 0 ? ni : Math.min(idx, L.length - 1);
    render();
  });
  const offFav = store.on('fav', updateBars);

  render();
}
