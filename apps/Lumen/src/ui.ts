import { t } from './i18n';

type Kid = Node | string | null | undefined | false;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props?: Record<string, unknown> | null,
  ...kids: Kid[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === 'class') el.className = String(v);
      else if (k === 'style' && typeof v === 'string') el.style.cssText = v;
      else if (k === 'text') el.textContent = String(v);
      else if (k === 'html') el.innerHTML = String(v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v as EventListener);
      else if (k in el) (el as unknown as Record<string, unknown>)[k] = v;
      else el.setAttribute(k, String(v));
    }
  }
  for (const kid of kids) if (kid !== null && kid !== undefined && kid !== false) el.append(kid);
  return el;
}

// ------------------------------------------------------------------ иконки

const P: Record<string, string> = {
  back: '<path d="M15 18l-6-6 6-6"/>',
  close: '<path d="M18 6L6 18M6 6l12 12"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
  more: '<circle cx="12" cy="5" r="1.6" fill="currentColor"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/><circle cx="12" cy="19" r="1.6" fill="currentColor"/>',
  share: '<path d="M4 12v7a2 2 0 002 2h12a2 2 0 002-2v-7M16 6l-4-4-4 4M12 2v13"/>',
  trash: '<path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6M10 11v6M14 11v6"/>',
  star: '<path d="M12 2l3.1 6.3 6.9 1-5 4.9 1.2 6.8L12 17.8 5.8 21l1.2-6.8-5-4.9 6.9-1z"/>',
  starFill: '<path fill="currentColor" d="M12 2l3.1 6.3 6.9 1-5 4.9 1.2 6.8L12 17.8 5.8 21l1.2-6.8-5-4.9 6.9-1z"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
  folder: '<path d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/>',
  albums: '<path d="M12 2l10 5-10 5L2 7zM2 17l10 5 10-5M2 12l10 5 10-5"/>',
  play: '<path fill="currentColor" stroke="none" d="M8 5v14l11-7z"/>',
  check: '<path d="M5 12l5 5L20 7"/>',
  move: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  copy: '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  settings: '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
  sort: '<path d="M3 6h18M6 12h12M10 18h4"/>',
  edit: '<path d="M12 20h9M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4z"/>',
  restore: '<path d="M3 12a9 9 0 109-9 9 9 0 00-6.7 3L3 8M3 3v5h5"/>',
  chevron: '<path d="M9 6l6 6-6 6"/>',
  eyeOff: '<path d="M17.9 17.9A10.9 10.9 0 0112 20c-7 0-11-8-11-8a18.5 18.5 0 015.1-5.9M9.9 4.2A9.1 9.1 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.2 3.2M1 1l22 22"/>',
  open: '<path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"/>',
};

export function icon(name: keyof typeof P | string, size = 24): SVGElement {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 24 24');
  s.setAttribute('width', String(size));
  s.setAttribute('height', String(size));
  s.setAttribute('fill', 'none');
  s.setAttribute('stroke', 'currentColor');
  s.setAttribute('stroke-width', '1.9');
  s.setAttribute('stroke-linecap', 'round');
  s.setAttribute('stroke-linejoin', 'round');
  s.setAttribute('aria-hidden', 'true');
  s.innerHTML = P[name] ?? '';
  return s;
}

export function iconBtn(name: string, label: string, onClick: (e: Event) => void, cls = ''): HTMLButtonElement {
  return h('button', { class: `ib ${cls}`, 'aria-label': label, title: label, onClick }, icon(name));
}

// ------------------------------------------------------------------ аппаратная «назад»

const backStack: (() => void)[] = [];

export function pushBack(fn: () => void): () => void {
  backStack.push(fn);
  return () => {
    const i = backStack.lastIndexOf(fn);
    if (i >= 0) backStack.splice(i, 1);
  };
}

export function handleBack(): boolean {
  const f = backStack.pop();
  if (!f) return false;
  f();
  return true;
}

// ------------------------------------------------------------------ оверлеи

const overlays = (): HTMLElement => document.getElementById('overlays')!;

export interface SheetHandle {
  el: HTMLElement;
  close(): void;
}

export function sheet(title: string | null, body: Node, opts: { full?: boolean; onClose?: () => void } = {}): SheetHandle {
  const scrim = h('div', { class: 'scrim' });
  const panel = h(
    'div',
    { class: `sheet${opts.full ? ' full' : ''}`, role: 'dialog' },
    h('div', { class: 'grip' }),
    title ? h('div', { class: 'sheet-title', text: title }) : null,
    h('div', { class: 'sheet-body' }, body)
  );
  const root = h('div', { class: 'layer' }, scrim, panel);
  overlays().append(root);
  requestAnimationFrame(() => root.classList.add('on'));

  let closed = false;
  let unback = (): void => undefined;
  const close = (): void => {
    if (closed) return;
    closed = true;
    unback();
    root.classList.remove('on');
    setTimeout(() => root.remove(), 220);
    opts.onClose?.();
  };
  unback = pushBack(close);
  scrim.addEventListener('click', close);

  // свайп вниз по «ручке» закрывает
  let y0 = 0;
  const grip = panel.querySelector('.grip') as HTMLElement;
  grip.addEventListener('pointerdown', (e) => {
    y0 = e.clientY;
    grip.setPointerCapture(e.pointerId);
  });
  grip.addEventListener('pointermove', (e) => {
    if (!grip.hasPointerCapture(e.pointerId)) return;
    const dy = Math.max(0, e.clientY - y0);
    panel.style.transform = `translateY(${dy}px)`;
  });
  grip.addEventListener('pointerup', (e) => {
    const dy = e.clientY - y0;
    panel.style.transform = '';
    if (dy > 80) close();
  });
  return { el: panel, close };
}

export interface MenuItem {
  icon?: string;
  label: string;
  hint?: string;
  danger?: boolean;
  active?: boolean;
  onClick: () => void;
}

export function menu(title: string | null, items: (MenuItem | null)[], onClose?: () => void): SheetHandle {
  const list = h('div', { class: 'menu' });
  const sh = sheet(title, list, { onClose });
  for (const it of items) {
    if (!it) {
      list.append(h('div', { class: 'menu-sep' }));
      continue;
    }
    list.append(
      h(
        'button',
        {
          class: `menu-item${it.danger ? ' danger' : ''}${it.active ? ' active' : ''}`,
          onClick: () => {
            sh.close();
            it.onClick();
          },
        },
        it.icon ? icon(it.icon, 22) : h('span', { class: 'menu-ic' }),
        h('span', { class: 'menu-label' }, it.label, it.hint ? h('small', { text: it.hint }) : null),
        it.active ? icon('check', 20) : null
      )
    );
  }
  return sh;
}

function dialog(build: (done: (v: unknown) => void) => HTMLElement): Promise<unknown> {
  return new Promise((resolve) => {
    let finished = false;
    let unback = (): void => undefined;
    const root = h('div', { class: 'layer dlg' });
    const finish = (v: unknown): void => {
      if (finished) return;
      finished = true;
      unback();
      root.classList.remove('on');
      setTimeout(() => root.remove(), 160);
      resolve(v);
    };
    const scrim = h('div', { class: 'scrim', onClick: () => finish(null) });
    root.append(scrim, build(finish));
    overlays().append(root);
    requestAnimationFrame(() => root.classList.add('on'));
    unback = pushBack(() => finish(null));
  });
}

export async function confirmDialog(title: string, text: string, okLabel: string, danger = false): Promise<boolean> {
  const r = await dialog((done) =>
    h(
      'div',
      { class: 'dialog', role: 'alertdialog' },
      h('h3', { text: title }),
      text ? h('p', { text }) : null,
      h(
        'div',
        { class: 'dialog-actions' },
        h('button', { class: 'btn text', text: t('cancel'), onClick: () => done(false) }),
        h('button', { class: `btn ${danger ? 'danger' : 'primary'}`, text: okLabel, onClick: () => done(true) })
      )
    )
  );
  return r === true;
}

export async function promptDialog(title: string, value: string, placeholder = '', okLabel = t('ok')): Promise<string | null> {
  const r = await dialog((done) => {
    const input = h('input', { class: 'field', type: 'text', value, placeholder, autocapitalize: 'off', spellcheck: false });
    const go = (): void => done(input.value);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') go();
    });
    setTimeout(() => {
      input.focus();
      const dot = value.lastIndexOf('.');
      input.setSelectionRange(0, dot > 0 ? dot : value.length);
    }, 60);
    return h(
      'div',
      { class: 'dialog' },
      h('h3', { text: title }),
      input,
      h(
        'div',
        { class: 'dialog-actions' },
        h('button', { class: 'btn text', text: t('cancel'), onClick: () => done(null) }),
        h('button', { class: 'btn primary', text: okLabel, onClick: go })
      )
    );
  });
  return typeof r === 'string' ? r : null;
}

let toastTimer: number | undefined;
export function toast(msg: string, ms = 2400): void {
  document.querySelectorAll('.toast').forEach((n) => n.remove());
  const el = h('div', { class: 'toast', role: 'status', text: msg });
  overlays().append(el);
  requestAnimationFrame(() => el.classList.add('on'));
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    el.classList.remove('on');
    setTimeout(() => el.remove(), 200);
  }, ms);
}

export function haptic(): void {
  void import('@capacitor/haptics')
    .then((m) => m.Haptics.impact({ style: m.ImpactStyle.Light }))
    .catch(() => undefined);
}
