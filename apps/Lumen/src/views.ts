import { api } from './native';
import { store, rowToItem } from './store';
import { thumbs } from './thumbs';
import { VList, type Row } from './vlist';
import { fmtCount, fmtDay, fmtDuration, fmtMonth, fmtSize, t } from './i18n';
import { h, haptic, icon, iconBtn, menu, sheet, pushBack } from './ui';
import {
  deleteForever,
  hideFolder,
  moveItems,
  restoreItems,
  shareItems,
  toggleFavorite,
  trashItems,
} from './actions';
import { openViewer } from './viewer';
import type { Folder, Item } from './types';

type Tab = 'photos' | 'albums' | 'folders' | 'favorites';

const GAP = 2;

export class App {
  private tab: Tab = 'photos';
  private folderPath = ''; // для вкладки «Папки»
  private albumPath: string | null = null; // открытый альбом
  private searchOn = false;
  private query = '';
  private trashMode = false;
  private trashItems: Item[] = [];

  private sel = new Set<string>();
  private selecting = false;

  /** плоский список медиа текущего экрана — по нему листает просмотрщик */
  private flat: Item[] = [];

  private root: HTMLElement;
  private topbar: HTMLElement;
  private title: HTMLElement;
  private actions: HTMLElement;
  private searchBox: HTMLInputElement;
  private selbar: HTMLElement;
  private selCount: HTMLElement;
  private nav: HTMLElement;
  private body: HTMLElement;
  private vl: VList;
  private note: HTMLElement;
  private unbackSel: (() => void) | null = null;
  private favBtnSel!: HTMLElement;

  constructor(mount: HTMLElement) {
    this.vl = new VList(
      (row, i) => this.renderRow(row, i),
      (ts) => fmtMonth(ts)
    );
    this.title = h('div', { class: 'tb-title' });
    this.actions = h('div', { class: 'tb-actions' });
    this.searchBox = h('input', {
      class: 'search',
      type: 'search',
      placeholder: t('search'),
      enterkeyhint: 'search',
      onInput: () => {
        this.query = this.searchBox.value;
        this.rebuild();
      },
    });
    this.topbar = h('header', { class: 'topbar' }, this.title, this.actions);
    this.selCount = h('div', { class: 'tb-title' });
    this.selbar = h('header', { class: 'topbar selbar' });
    this.nav = h('nav', { class: 'bottomnav' });
    this.note = h('div', { class: 'center-note' });
    this.body = h('main', { class: 'content' }, this.vl.scroller, this.vl.fastEl, this.note);
    this.root = h('div', { class: 'app' }, this.topbar, this.selbar, this.body, this.nav);
    mount.append(this.root);

    this.buildNav();
    this.bindSelectionGestures();
    store.on('change', () => this.rebuild(true));
    store.on('state', () => this.rebuild(true));
    store.on('fav', () => {
      if (this.tab === 'favorites') this.rebuild(true);
      else this.vl.eachLive((el) => this.syncCells(el));
    });
    window.addEventListener('resize', () => this.rebuild(true));
    this.rebuild();
  }

  // ------------------------------------------------------------------ навигация

  private buildNav(): void {
    const defs: [Tab, string, string][] = [
      ['photos', 'image', 'photos'],
      ['albums', 'albums', 'albums'],
      ['folders', 'folder', 'folders'],
      ['favorites', 'star', 'favorites'],
    ];
    this.nav.textContent = '';
    for (const [id, ic, label] of defs) {
      this.nav.append(
        h(
          'button',
          {
            class: `nav-item${this.tab === id && !this.trashMode ? ' on' : ''}`,
            onClick: () => this.setTab(id),
          },
          icon(ic, 24),
          h('span', { text: t(label) })
        )
      );
    }
  }

  private setTab(tab: Tab): void {
    if (this.selecting) this.clearSel();
    const same = this.tab === tab && !this.trashMode;
    this.trashMode = false;
    thumbs.setTrashMode(false);
    this.tab = tab;
    if (same) {
      // повторный тап — наверх / на уровень выше
      if (tab === 'albums' && this.albumPath !== null) this.albumPath = null;
      else if (tab === 'folders' && this.folderPath) this.folderPath = store.folders.get(this.folderPath)?.parent ?? '';
      else {
        this.vl.scroller.scrollTo({ top: 0, behavior: 'smooth' });
        return;
      }
    } else {
      this.albumPath = null;
    }
    this.searchOn = false;
    this.query = '';
    this.buildNav();
    this.rebuild();
  }

  /** Аппаратная кнопка «назад». true — обработали. */
  back(): boolean {
    if (this.selecting) {
      this.clearSel();
      return true;
    }
    if (this.searchOn) {
      this.searchOn = false;
      this.query = '';
      this.rebuild();
      return true;
    }
    if (this.trashMode) {
      this.trashMode = false;
      thumbs.setTrashMode(false);
      this.buildNav();
      this.rebuild();
      return true;
    }
    if (this.tab === 'albums' && this.albumPath !== null) {
      this.albumPath = null;
      this.rebuild();
      return true;
    }
    if (this.tab === 'folders' && this.folderPath) {
      this.folderPath = store.folders.get(this.folderPath)?.parent ?? '';
      this.rebuild();
      return true;
    }
    if (this.tab !== 'photos') {
      this.setTab('photos');
      return true;
    }
    return false;
  }

  // ------------------------------------------------------------------ шапка

  private renderTopbar(): void {
    this.actions.textContent = '';
    this.topbar.classList.toggle('hidden', this.selecting);
    this.selbar.classList.toggle('hidden', !this.selecting);

    const canBack =
      this.trashMode ||
      (this.tab === 'albums' && this.albumPath !== null) ||
      (this.tab === 'folders' && this.folderPath !== '');

    this.title.textContent = '';
    if (this.searchOn) {
      this.title.append(this.searchBox);
      this.actions.append(
        iconBtn('close', t('close'), () => {
          this.searchOn = false;
          this.query = '';
          this.rebuild();
        })
      );
      setTimeout(() => this.searchBox.focus(), 30);
      return;
    }

    let label = t(this.tab);
    if (this.trashMode) label = t('trash');
    else if (this.tab === 'albums' && this.albumPath !== null) label = store.folders.get(this.albumPath)?.name ?? '';
    else if (this.tab === 'folders') label = this.folderPath ? store.folders.get(this.folderPath)?.name ?? '' : t('folders');

    if (canBack) this.title.append(iconBtn('back', t('back'), () => void this.back()));
    this.title.append(h('span', { class: 'tb-label', text: label }));

    if (!this.trashMode) this.actions.append(iconBtn('search', t('search'), () => this.openSearch()));
    if (!this.trashMode && (this.tab === 'photos' || this.tab === 'favorites' || this.albumPath !== null || this.tab === 'folders'))
      this.actions.append(iconBtn('sort', t('sort'), () => this.openViewMenu()));
    this.actions.append(iconBtn('more', t('more'), () => this.openMainMenu()));
  }

  private renderSelbar(): void {
    this.selbar.textContent = '';
    this.selCount.textContent = t('selected', { n: this.sel.size });
    const items = this.selItems();
    const allFav = items.length > 0 && items.every((i) => store.fav.has(i.k));
    this.favBtnSel = iconBtn(allFav ? 'starFill' : 'star', t('favorite'), () => {
      toggleFavorite(this.selItems());
      this.clearSel();
    });
    const trashSel = this.trashMode;
    this.selbar.append(
      iconBtn('close', t('close'), () => this.clearSel()),
      this.selCount,
      h(
        'div',
        { class: 'tb-actions' },
        ...(trashSel
          ? [
              iconBtn('restore', t('restore'), async () => {
                const it = this.selItems();
                if (await restoreItems(it)) await this.reloadTrash();
                this.clearSel();
              }),
              iconBtn('trash', t('deleteForever'), async () => {
                const it = this.selItems();
                if (await deleteForever(it)) await this.reloadTrash();
                this.clearSel();
              }),
            ]
          : [
              iconBtn('share', t('share'), () => void shareItems(this.selItems())),
              this.favBtnSel,
              iconBtn('move', t('move'), async () => {
                if (await moveItems(this.selItems(), false)) this.clearSel();
              }),
              iconBtn('trash', t('delete'), async () => {
                if (await trashItems(this.selItems())) this.clearSel();
              }),
              iconBtn('more', t('more'), () => this.openSelMenu()),
            ])
      )
    );
  }

  // ------------------------------------------------------------------ меню

  private openSearch(): void {
    this.searchOn = true;
    this.tab = this.tab === 'albums' || this.tab === 'folders' ? 'photos' : this.tab;
    this.albumPath = null;
    this.buildNav();
    this.rebuild();
  }

  private openMainMenu(): void {
    menu(null, [
      { icon: 'restore', label: t('trash'), onClick: () => void this.openTrash() },
      { icon: 'settings', label: t('settings'), onClick: () => this.openSettings() },
    ]);
  }

  private openSelMenu(): void {
    menu(null, [
      { icon: 'check', label: t('selectAll'), onClick: () => this.selectAll() },
      { icon: 'copy', label: t('copy'), onClick: async () => void ((await moveItems(this.selItems(), true)) && this.clearSel()) },
    ]);
  }

  private openViewMenu(): void {
    const s = store.settings;
    menu(t('sort'), [
      { label: t('sortDate'), active: s.sort === 'date', onClick: () => store.update({ sort: 'date' }) },
      { label: t('sortName'), active: s.sort === 'name', onClick: () => store.update({ sort: 'name' }) },
      { label: t('sortSize'), active: s.sort === 'size', onClick: () => store.update({ sort: 'size' }) },
      null,
      { label: t('desc'), active: s.desc, onClick: () => store.update({ desc: true }) },
      { label: t('asc'), active: !s.desc, onClick: () => store.update({ desc: false }) },
      null,
      { label: `${t('filter')}: ${t('all')}`, active: s.filter === 'all', onClick: () => store.update({ filter: 'all' }) },
      { label: `${t('filter')}: ${t('onlyPhotos')}`, active: s.filter === 'photo', onClick: () => store.update({ filter: 'photo' }) },
      { label: `${t('filter')}: ${t('onlyVideos')}`, active: s.filter === 'video', onClick: () => store.update({ filter: 'video' }) },
    ]);
  }

  private openSettings(): void {
    const s = store.settings;
    const body = h('div', { class: 'settings' });
    let sh: ReturnType<typeof sheet>;

    const seg = <T extends string>(label: string, value: T, opts: [T, string][], set: (v: T) => void): void => {
      const row = h('div', { class: 'seg' });
      for (const [v, text] of opts) {
        row.append(
          h('button', {
            class: v === value ? 'on' : '',
            text,
            onClick: (e: Event) => {
              set(v);
              row.querySelectorAll('button').forEach((b) => b.classList.remove('on'));
              (e.currentTarget as HTMLElement).classList.add('on');
            },
          })
        );
      }
      body.append(h('div', { class: 'set-label', text: label }), row);
    };
    const toggle = (label: string, value: boolean, set: (v: boolean) => void): void => {
      const input = h('input', { type: 'checkbox', checked: value, onChange: () => set(input.checked) });
      body.append(h('label', { class: 'set-row' }, h('span', { text: label }), h('span', { class: 'switch' }, input, h('i'))));
    };

    seg(t('theme'), s.theme, [
      ['auto', t('themeAuto')],
      ['dark', t('themeDark')],
      ['light', t('themeLight')],
      ['amoled', t('themeAmoled')],
    ], (v) => store.update({ theme: v }));

    const colLabel = h('div', { class: 'set-label', text: `${t('columns')}: ${s.cols}` });
    const range = h('input', {
      type: 'range',
      min: 2,
      max: 7,
      step: 1,
      value: String(s.cols),
      class: 'range',
      onInput: () => {
        colLabel.textContent = `${t('columns')}: ${range.value}`;
        store.update({ cols: Number(range.value) });
      },
    });
    body.append(colLabel, range);

    toggle(t('groupByDay'), s.groupByDay, (v) => store.update({ groupByDay: v }));
    toggle(t('fitPhotos'), s.fit, (v) => store.update({ fit: v }));

    seg(t('language'), s.lang, [
      ['auto', t('langAuto')],
      ['ru', 'Русский'],
      ['en', 'English'],
    ], (v) => {
      store.update({ lang: v });
      sh.close();
      this.buildNav();
      this.rebuild();
    });

    // MANAGE_MEDIA
    const mm = h('div', { class: 'set-row mm' });
    body.append(mm);
    void api.manageMediaStatus().then((st) => {
      if (!st.supported) return;
      mm.append(
        h('span', {}, t('manageMedia'), h('small', { text: t('manageMediaHint') })),
        h('button', {
          class: 'btn tonal',
          text: st.granted ? t('manageMediaOn') : t('manageMediaOff'),
          disabled: st.granted,
          onClick: () => void api.requestManageMedia(),
        })
      );
    });

    // скрытые папки
    body.append(h('div', { class: 'set-label', text: t('hiddenFolders') }));
    if (!s.hidden.length) body.append(h('div', { class: 'muted', text: t('noHidden') }));
    for (const p of s.hidden) {
      body.append(
        h(
          'div',
          { class: 'set-row' },
          h('span', { text: p.replace(/\/$/, '') }),
          h('button', {
            class: 'btn text',
            text: t('unhide'),
            onClick: (e: Event) => {
              store.update({ hidden: store.settings.hidden.filter((x) => x !== p) });
              (e.currentTarget as HTMLElement).closest('.set-row')!.remove();
            },
          })
        )
      );
    }
    body.append(h('div', { class: 'about', text: `${t('about')} · v1.0` }));
    sh = sheet(t('settings'), body, { full: true });
  }

  // ------------------------------------------------------------------ корзина

  private async openTrash(): Promise<void> {
    this.trashMode = true;
    thumbs.setTrashMode(true);
    this.searchOn = false;
    this.buildNav();
    await this.reloadTrash();
  }

  private async reloadTrash(): Promise<void> {
    try {
      const { rows } = await api.listTrash();
      this.trashItems = rows.map(rowToItem);
    } catch {
      this.trashItems = [];
    }
    thumbs.forget(this.trashItems.map((i) => i.k));
    this.rebuild();
  }

  // ------------------------------------------------------------------ выделение

  private selItems(): Item[] {
    const pool = this.trashMode ? this.trashItems : store.items;
    const map = new Map(pool.map((i) => [i.k, i]));
    return [...this.sel].map((k) => map.get(k)).filter((x): x is Item => !!x);
  }

  private setSelecting(on: boolean): void {
    if (on === this.selecting) return;
    this.selecting = on;
    this.root.classList.toggle('selecting', on);
    if (on) this.unbackSel = pushBack(() => this.clearSel());
    else {
      this.unbackSel?.();
      this.unbackSel = null;
    }
  }

  private toggleSel(k: string, force?: boolean): void {
    const on = force ?? !this.sel.has(k);
    if (on) this.sel.add(k);
    else this.sel.delete(k);
    if (!this.sel.size) this.clearSel();
    else {
      this.setSelecting(true);
      this.renderTopbar();
      this.renderSelbar();
      this.vl.eachLive((el) => this.syncCells(el));
    }
  }

  private selectAll(): void {
    for (const i of this.flat) this.sel.add(i.k);
    this.setSelecting(true);
    this.renderTopbar();
    this.renderSelbar();
    this.vl.eachLive((el) => this.syncCells(el));
  }

  private clearSel(): void {
    this.sel.clear();
    this.setSelecting(false);
    this.renderTopbar();
    this.vl.eachLive((el) => this.syncCells(el));
  }

  private syncCells(el: HTMLElement): void {
    el.querySelectorAll<HTMLElement>('.cell').forEach((c) => {
      const k = c.dataset.k!;
      c.classList.toggle('sel', this.sel.has(k));
      c.classList.toggle('fav', store.fav.has(k));
    });
  }

  /** Долгое нажатие + протяжка пальцем = выделение диапазона. */
  private bindSelectionGestures(): void {
    const sc = this.vl.scroller;
    let timer = 0;
    let startX = 0;
    let startY = 0;
    let dragging = false;
    let dragMode = true;
    let suppressClick = false;

    const cellAt = (x: number, y: number): HTMLElement | null =>
      (document.elementFromPoint(x, y) as HTMLElement | null)?.closest<HTMLElement>('.cell') ?? null;

    sc.addEventListener('pointerdown', (e) => {
      const c = (e.target as HTMLElement).closest<HTMLElement>('.cell');
      if (!c) return;
      startX = e.clientX;
      startY = e.clientY;
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        dragging = true;
        suppressClick = true;
        haptic();
        const k = c.dataset.k!;
        dragMode = !this.sel.has(k);
        this.toggleSel(k, dragMode);
      }, 380);
    });
    sc.addEventListener('pointermove', (e) => {
      if (!dragging) {
        if (Math.hypot(e.clientX - startX, e.clientY - startY) > 9) window.clearTimeout(timer);
        return;
      }
      const c = cellAt(e.clientX, e.clientY);
      if (c && c.dataset.k && this.sel.has(c.dataset.k) !== dragMode) this.toggleSel(c.dataset.k, dragMode);
      const r = sc.getBoundingClientRect();
      if (e.clientY < r.top + 90) sc.scrollTop -= 14;
      else if (e.clientY > r.bottom - 90) sc.scrollTop += 14;
    });
    const end = (): void => {
      window.clearTimeout(timer);
      dragging = false;
      setTimeout(() => {
        suppressClick = false;
      }, 60);
    };
    sc.addEventListener('pointerup', end);
    sc.addEventListener('pointercancel', end);
    sc.addEventListener(
      'touchmove',
      (e) => {
        if (dragging) e.preventDefault();
      },
      { passive: false }
    );
    sc.addEventListener('contextmenu', (e) => e.preventDefault());

    sc.addEventListener(
      'click',
      (e) => {
        if (suppressClick) {
          e.stopPropagation();
          return;
        }
        const target = e.target as HTMLElement;
        const cell = target.closest<HTMLElement>('.cell');
        if (cell) {
          const k = cell.dataset.k!;
          if (this.selecting) this.toggleSel(k);
          else this.openItem(k);
          return;
        }
        if (target.closest('button')) return;
        const fo = target.closest<HTMLElement>('[data-folder]');
        if (fo) this.openFolder(fo.dataset.folder!);
      },
      true
    );
  }

  private openItem(k: string): void {
    if (this.trashMode) return;
    const i = this.flat.findIndex((x) => x.k === k);
    if (i < 0) return;
    openViewer(this.flat, i, (last) => {
      if (last) this.scrollToItem(last.k);
    });
  }

  private scrollToItem(k: string): void {
    const sc = this.vl.scroller;
    let y = 0;
    for (const r of this.rows) {
      if (r.kind === 'cells' && (r.data as { items: Item[] }).items.some((i) => i.k === k)) {
        const vh = sc.clientHeight;
        if (y < sc.scrollTop || y + r.h > sc.scrollTop + vh) sc.scrollTop = Math.max(0, y - vh / 2 + r.h / 2);
        return;
      }
      y += r.h;
    }
  }

  private openFolder(path: string): void {
    if (this.tab === 'albums') this.albumPath = path;
    else this.folderPath = path;
    this.rebuild();
  }

  // ------------------------------------------------------------------ построение строк

  private rows: Row[] = [];

  private colsCount(): number {
    const w = window.innerWidth;
    const base = store.settings.cols;
    return w > 700 ? Math.round(base * (w / 400)) : base;
  }

  private mediaRows(list: Item[], group: boolean): Row[] {
    const rows: Row[] = [];
    const width = this.vl.scroller.clientWidth || window.innerWidth;
    const cols = this.colsCount();
    const cell = (width - GAP * (cols - 1)) / cols;
    const rowH = cell + GAP;
    const fit = store.settings.fit;
    const sameDay = (a: number, b: number): boolean => new Date(a).toDateString() === new Date(b).toDateString();

    let i = 0;
    while (i < list.length) {
      let end = i;
      if (group) {
        while (end < list.length && sameDay(list[end].taken, list[i].taken)) end++;
        rows.push({ h: 44, kind: 'head', data: fmtDay(list[i].taken) + ` · ${end - i}`, ts: list[i].taken });
      } else end = list.length;
      for (let j = i; j < end; j += cols) {
        rows.push({ h: rowH, kind: 'cells', data: { items: list.slice(j, Math.min(end, j + cols)), cell, fit }, ts: list[j].taken });
      }
      i = end;
    }
    return rows;
  }

  private folderRows(path: string): { rows: Row[]; media: Item[] } {
    const f = store.folders.get(path);
    const rows: Row[] = [];
    if (!f) return { rows, media: [] };
    const subs = f.children
      .map((p) => store.folders.get(p)!)
      .filter(Boolean)
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    for (const s of subs) rows.push({ h: 68, kind: 'folder', data: s });
    const media = f.direct;
    if (media.length) rows.push(...this.mediaRows(media, false));
    return { rows, media };
  }

  private albumRows(): Row[] {
    const albums = store.albums();
    const width = this.vl.scroller.clientWidth || window.innerWidth;
    const per = width > 700 ? 4 : 2;
    const pad = 12;
    const cw = (width - pad * (per + 1)) / per;
    const rows: Row[] = [{ h: 8, kind: 'pad' }];
    for (let i = 0; i < albums.length; i += per) {
      rows.push({ h: cw + 52, kind: 'albums', data: { folders: albums.slice(i, i + per), cw, pad } });
    }
    return rows;
  }

  rebuild(keepScroll = false): void {
    this.buildNav();
    this.renderTopbar();
    if (this.selecting) this.renderSelbar();

    // фокус в строке поиска не должен теряться при перестройке
    let rows: Row[] = [];
    let noteText = '';
    this.flat = [];

    if (store.state === 'init' || store.state === 'loading') {
      noteText = t('scanning');
    } else if (store.state === 'denied') {
      this.showDenied();
      return;
    } else if (this.trashMode) {
      this.flat = this.trashItems;
      rows = this.mediaRows(this.trashItems, false);
      noteText = this.trashItems.length ? '' : t('emptyTrash');
    } else if (this.searchOn) {
      this.flat = store.search(this.query);
      rows = this.mediaRows(this.flat, false);
      noteText = this.query.trim() && !this.flat.length ? t('emptySearch') : '';
    } else if (this.tab === 'photos') {
      this.flat = store.items;
      rows = this.mediaRows(this.flat, store.settings.groupByDay && store.settings.sort === 'date');
      noteText = this.flat.length ? '' : t('emptyLibrary');
    } else if (this.tab === 'favorites') {
      this.flat = store.favItems();
      rows = this.mediaRows(this.flat, false);
      noteText = this.flat.length ? '' : t('emptyFav');
    } else if (this.tab === 'albums') {
      if (this.albumPath === null) rows = this.albumRows();
      else {
        this.flat = store.folders.get(this.albumPath)?.direct ?? [];
        rows = this.mediaRows(this.flat, false);
      }
      noteText = rows.length > 1 || this.flat.length ? '' : t('emptyLibrary');
    } else {
      const { rows: r, media } = this.folderRows(this.folderPath);
      rows = r;
      this.flat = media;
      noteText = r.length ? '' : t('emptyLibrary');
    }

    this.rows = rows;
    this.note.textContent = noteText;
    this.note.classList.toggle('on', !!noteText);
    this.vl.setRows(rows, keepScroll);
  }

  private showDenied(): void {
    this.rows = [];
    this.vl.setRows([]);
    this.note.classList.add('on');
    this.note.textContent = '';
    this.note.append(
      h('div', { class: 'denied' }, icon('image', 56), h('h2', { text: t('noAccessTitle') }), h('p', { text: t('noAccessText') }), h('button', { class: 'btn primary', text: t('grant'), onClick: () => void store.load() }), h('small', { text: t('openSettings') }))
    );
  }

  // ------------------------------------------------------------------ рендер строк

  private renderRow(row: Row, _i: number): HTMLElement {
    switch (row.kind) {
      case 'head':
        return h('div', { class: 'day-head' }, h('span', { text: String(row.data) }));
      case 'pad':
        return h('div');
      case 'folder':
        return this.folderRow(row.data as Folder);
      case 'albums':
        return this.albumsRow(row.data as { folders: Folder[]; cw: number; pad: number });
      case 'cells':
        return this.cellsRow(row.data as { items: Item[]; cell: number; fit: boolean });
      default:
        return h('div');
    }
  }

  private cellsRow(d: { items: Item[]; cell: number; fit: boolean }): HTMLElement {
    const el = h('div', { class: 'cells' });
    d.items.forEach((it, n) => {
      el.append(this.cell(it, d.cell, n * (d.cell + GAP), d.fit));
    });
    return el;
  }

  private cell(it: Item, size: number, left: number, fit: boolean): HTMLElement {
    const img = h('img', { class: fit ? 'fit' : '', alt: '', draggable: false, decoding: 'async' });
    thumbs.attach(img, it);
    const c = h(
      'div',
      {
        class: `cell${this.sel.has(it.k) ? ' sel' : ''}${store.fav.has(it.k) ? ' fav' : ''}${it.v ? ' video' : ''}`,
        style: `width:${size}px;height:${size}px;left:${left}px`,
        'data-k': it.k,
      },
      img,
      it.v ? h('span', { class: 'badge-v' }, icon('play', 12), it.dur ? fmtDuration(it.dur) : '') : null,
      h('span', { class: 'badge-f' }, icon('starFill', 14)),
      h('span', { class: 'check' }, icon('check', 16))
    );
    return c;
  }

  private folderRow(f: Folder): HTMLElement {
    const th = h('div', { class: 'folder-thumb' });
    if (f.cover) {
      const img = h('img', { alt: '', draggable: false });
      thumbs.attach(img, f.cover);
      th.append(img);
    } else th.append(icon('folder', 26));
    return h(
      'div',
      { class: 'folder-row', 'data-folder': f.path },
      th,
      h('div', { class: 'folder-info' }, h('div', { class: 'folder-name', text: f.name }), h('div', { class: 'folder-meta', text: t('folderStat', { c: fmtCount(f.count), s: fmtSize(f.size) }) })),
      h(
        'button',
        {
          class: 'ib small',
          'aria-label': t('more'),
          onClick: (e: Event) => {
            e.stopPropagation();
            menu(f.name, [
              { icon: 'eyeOff', label: t('hideFolder'), onClick: () => hideFolder(f.path) },
              { icon: 'check', label: t('selectAll'), onClick: () => this.selectFolder(f) },
            ]);
          },
        },
        icon('more', 20)
      )
    );
  }

  private selectFolder(f: Folder): void {
    const collect = (p: string): void => {
      const x = store.folders.get(p);
      if (!x) return;
      x.direct.forEach((i) => this.sel.add(i.k));
      x.children.forEach(collect);
    };
    collect(f.path);
    this.setSelecting(true);
    this.renderTopbar();
    this.renderSelbar();
  }

  private albumsRow(d: { folders: Folder[]; cw: number; pad: number }): HTMLElement {
    const el = h('div', { class: 'albums-row' });
    d.folders.forEach((f, n) => {
      const img = h('img', { alt: '', draggable: false });
      if (f.cover) thumbs.attach(img, f.cover);
      el.append(
        h(
          'div',
          {
            class: 'album',
            'data-folder': f.path,
            style: `width:${d.cw}px;left:${d.pad + n * (d.cw + d.pad)}px`,
          },
          h('div', { class: 'album-cover', style: `height:${d.cw}px` }, img),
          h('div', { class: 'album-name', text: f.name }),
          h('div', { class: 'album-count', text: fmtCount(f.direct.length) })
        )
      );
    });
    return el;
  }
}

