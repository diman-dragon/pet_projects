/** Виртуальный список разнородных строк. Держит в DOM только то, что на экране. */

export interface Row {
  h: number;
  kind: 'head' | 'cells' | 'folder' | 'albums' | 'note' | 'pad';
  data?: unknown;
  /** время для подписи быстрой прокрутки */
  ts?: number;
}

export class VList {
  readonly scroller: HTMLElement;
  private inner: HTMLElement;
  private rows: Row[] = [];
  private offs: number[] = [];
  private total = 0;
  private live = new Map<number, HTMLElement>();
  private raf = 0;
  private fast: HTMLElement;
  private fastLabel: HTMLElement;
  private fastTimer = 0;
  private dragging = false;

  constructor(
    private render: (row: Row, index: number) => HTMLElement,
    private fmtLabel: (ts: number) => string
  ) {
    this.inner = document.createElement('div');
    this.inner.className = 'vinner';
    this.scroller = document.createElement('div');
    this.scroller.className = 'scroller';
    this.scroller.append(this.inner);

    this.fastLabel = document.createElement('span');
    this.fastLabel.className = 'fast-label';
    this.fast = document.createElement('div');
    this.fast.className = 'fast';
    this.fast.append(this.fastLabel);

    this.scroller.addEventListener('scroll', () => this.onScroll(), { passive: true });
    new ResizeObserver(() => this.schedule()).observe(this.scroller);
    this.bindFast();
  }

  get fastEl(): HTMLElement {
    return this.fast;
  }

  setRows(rows: Row[], keepScroll = false): void {
    const top = this.scroller.scrollTop;
    this.rows = rows;
    this.offs = new Array(rows.length);
    let y = 0;
    for (let i = 0; i < rows.length; i++) {
      this.offs[i] = y;
      y += rows[i].h;
    }
    this.total = y;
    this.inner.style.height = `${y}px`;
    for (const el of this.live.values()) el.remove();
    this.live.clear();
    this.scroller.scrollTop = keepScroll ? top : 0;
    this.update();
  }

  /** Перерисовать видимые строки (например, после смены выделения). */
  refresh(): void {
    for (const el of this.live.values()) el.remove();
    this.live.clear();
    this.update();
  }

  /** Вызвать для каждой живой строки (быстрое обновление состояния без перерисовки). */
  eachLive(fn: (el: HTMLElement, row: Row) => void): void {
    for (const [i, el] of this.live) fn(el, this.rows[i]);
  }

  private schedule(): void {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.update();
    });
  }

  private rowAt(y: number): number {
    let lo = 0;
    let hi = this.offs.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.offs[mid] <= y) lo = mid;
      else hi = mid - 1;
    }
    return Math.max(0, lo);
  }

  private update(): void {
    if (!this.rows.length) return;
    const top = this.scroller.scrollTop;
    const vh = this.scroller.clientHeight || window.innerHeight;
    const pad = vh * 0.8;
    const a = this.rowAt(Math.max(0, top - pad));
    const b = this.rowAt(top + vh + pad);
    for (const [i, el] of this.live) {
      if (i < a || i > b) {
        el.remove();
        this.live.delete(i);
      }
    }
    for (let i = a; i <= b && i < this.rows.length; i++) {
      if (this.live.has(i)) continue;
      const el = this.render(this.rows[i], i);
      el.classList.add('vrow');
      el.style.height = `${this.rows[i].h}px`;
      el.style.transform = `translateY(${this.offs[i]}px)`;
      this.inner.append(el);
      this.live.set(i, el);
    }
  }

  // --------------------------------------------------------------- быстрая прокрутка

  private onScroll(): void {
    this.schedule();
    if (this.total <= this.scroller.clientHeight * 3) return;
    this.placeThumb();
    this.fast.classList.add('on');
    window.clearTimeout(this.fastTimer);
    this.fastTimer = window.setTimeout(() => {
      if (!this.dragging) this.fast.classList.remove('on');
    }, 1100);
  }

  private placeThumb(): void {
    const max = this.total - this.scroller.clientHeight;
    const trackH = this.fast.clientHeight - 48;
    const r = max > 0 ? this.scroller.scrollTop / max : 0;
    this.fast.style.setProperty('--ty', `${r * trackH}px`);
    const row = this.rows[this.rowAt(this.scroller.scrollTop)];
    this.fastLabel.textContent = row?.ts ? this.fmtLabel(row.ts) : '';
  }

  private bindFast(): void {
    const move = (e: PointerEvent): void => {
      const rect = this.fast.getBoundingClientRect();
      const r = Math.min(1, Math.max(0, (e.clientY - rect.top - 24) / Math.max(1, rect.height - 48)));
      this.scroller.scrollTop = r * (this.total - this.scroller.clientHeight);
    };
    this.fast.addEventListener('pointerdown', (e) => {
      this.dragging = true;
      this.fast.setPointerCapture(e.pointerId);
      this.fast.classList.add('on', 'drag');
      move(e);
      e.preventDefault();
    });
    this.fast.addEventListener('pointermove', (e) => {
      if (this.dragging) move(e);
    });
    const end = (): void => {
      this.dragging = false;
      this.fast.classList.remove('drag');
      window.clearTimeout(this.fastTimer);
      this.fastTimer = window.setTimeout(() => this.fast.classList.remove('on'), 900);
    };
    this.fast.addEventListener('pointerup', end);
    this.fast.addEventListener('pointercancel', end);
  }
}
