import { Preferences } from '@capacitor/preferences';
import * as Core from '@capacitor/core';
import { api } from './native';
import { setLang, t } from './i18n';
import type { Folder, Item, Settings } from './types';

type Ev = 'change' | 'settings' | 'fav' | 'state';
type State = 'init' | 'denied' | 'loading' | 'ready';

const DEFAULTS: Settings = {
  theme: 'auto',
  cols: 4,
  sort: 'date',
  desc: true,
  filter: 'all',
  fit: false,
  groupByDay: true,
  hidden: [],
  lang: 'auto',
};

const CHUNK = 6000;

export function rowToItem(r: (string | number)[]): Item {
  const v = r[1] === 1;
  const id = r[0] as number;
  return {
    k: (v ? 'v' : 'i') + id,
    id,
    v,
    name: r[2] as string,
    dir: r[3] as string,
    taken: r[4] as number,
    added: r[5] as number,
    size: r[6] as number,
    w: r[7] as number,
    h: r[8] as number,
    mime: r[9] as string,
    dur: r[10] as number,
  };
}

class Store {
  state: State = 'init';
  all: Item[] = [];
  /** после фильтров (скрытые папки, тип) и сортировки */
  items: Item[] = [];
  byKey = new Map<string, Item>();
  folders = new Map<string, Folder>();
  fav = new Set<string>();
  settings: Settings = { ...DEFAULTS };
  private listeners = new Map<Ev, Set<() => void>>();
  private scanning = false;
  private again = false;

  on(ev: Ev, fn: () => void): () => void {
    let s = this.listeners.get(ev);
    if (!s) this.listeners.set(ev, (s = new Set()));
    s.add(fn);
    return () => s!.delete(fn);
  }

  private emit(ev: Ev): void {
    this.listeners.get(ev)?.forEach((f) => f());
  }

  // --------------------------------------------------------------- persistence

  async init(): Promise<void> {
    try {
      const s = await Preferences.get({ key: 'settings' });
      if (s.value) this.settings = { ...DEFAULTS, ...JSON.parse(s.value) };
      const f = await Preferences.get({ key: 'fav' });
      if (f.value) this.fav = new Set<string>(JSON.parse(f.value));
    } catch {
      /* первый запуск */
    }
    this.applyAppearance();
  }

  private persist(key: 'settings' | 'fav'): void {
    const value = key === 'settings' ? JSON.stringify(this.settings) : JSON.stringify([...this.fav]);
    void Preferences.set({ key, value });
  }

  applyAppearance(): void {
    setLang(this.settings.lang);
    const root = document.documentElement;
    if (this.settings.theme === 'auto') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', this.settings.theme);
    root.style.setProperty('--cols', String(this.settings.cols));
    const light =
      this.settings.theme === 'light' ||
      (this.settings.theme === 'auto' && window.matchMedia('(prefers-color-scheme: light)').matches);
    const bg = light ? '#f6f6f7' : this.settings.theme === 'amoled' ? '#000000' : '#0b0b0d';
    try {
      const sb = (Core as unknown as { SystemBars?: { setStyle(o: { style: string }): Promise<void> } }).SystemBars;
      void sb?.setStyle({ style: light ? 'LIGHT' : 'DARK' });
      void import('@capacitor/status-bar').then((m) => {
        void m.StatusBar.setBackgroundColor({ color: bg }).catch(() => undefined);
        void m.StatusBar.setStyle({ style: light ? m.Style.Light : m.Style.Dark }).catch(() => undefined);
      });
    } catch {
      /* web */
    }
  }

  update(patch: Partial<Settings>): void {
    this.settings = { ...this.settings, ...patch };
    this.persist('settings');
    this.applyAppearance();
    this.rebuild();
    this.emit('settings');
  }

  // --------------------------------------------------------------- favorites

  toggleFav(keys: string[]): boolean {
    const allFav = keys.every((k) => this.fav.has(k));
    for (const k of keys) {
      if (allFav) this.fav.delete(k);
      else this.fav.add(k);
    }
    this.persist('fav');
    this.emit('fav');
    return !allFav;
  }

  favItems(): Item[] {
    return this.items.filter((i) => this.fav.has(i.k));
  }

  // --------------------------------------------------------------- loading

  async load(): Promise<void> {
    this.state = 'loading';
    this.emit('state');
    try {
      let a = await api.checkAccess();
      if (!a.granted) a = await api.requestAccess();
      if (!a.granted) {
        this.state = 'denied';
        this.emit('state');
        return;
      }
    } catch {
      this.state = 'denied';
      this.emit('state');
      return;
    }
    await this.rescan();
  }

  async rescan(): Promise<void> {
    if (this.scanning) {
      this.again = true;
      return;
    }
    this.scanning = true;
    try {
      const { total } = await api.scan();
      const acc: Item[] = [];
      for (let off = 0; off < total; off += CHUNK) {
        const { rows } = await api.fetch({ offset: off, limit: CHUNK });
        for (const r of rows) acc.push(rowToItem(r));
        if (off === 0 && total > CHUNK) this.setAll(acc.slice());
      }
      this.setAll(acc);
    } finally {
      this.scanning = false;
      if (this.again) {
        this.again = false;
        void this.rescan();
      }
    }
  }

  private setAll(list: Item[]): void {
    this.all = list;
    this.state = 'ready';
    this.rebuild();
    this.emit('state');
  }

  removeKeys(keys: Iterable<string>): void {
    const gone = new Set(keys);
    this.all = this.all.filter((i) => !gone.has(i.k));
    let favChanged = false;
    for (const k of gone) if (this.fav.delete(k)) favChanged = true;
    if (favChanged) this.persist('fav');
    this.rebuild();
  }

  // --------------------------------------------------------------- derived data

  isHidden(dir: string): boolean {
    return this.settings.hidden.some((h) => dir.startsWith(h));
  }

  private rebuild(): void {
    const s = this.settings;
    let list = this.all.filter((i) => !this.isHidden(i.dir));
    if (s.filter === 'photo') list = list.filter((i) => !i.v);
    else if (s.filter === 'video') list = list.filter((i) => i.v);

    const dir = s.desc ? -1 : 1;
    if (s.sort === 'date') {
      // native уже отдаёт по дате по убыванию — не сортируем зря
      if (!s.desc) list = list.slice().reverse();
      else list = list.slice();
    } else if (s.sort === 'name') {
      list = list.slice().sort((a, b) => dir * a.name.localeCompare(b.name, undefined, { numeric: true }));
    } else {
      list = list.slice().sort((a, b) => dir * (a.size - b.size));
    }
    this.items = list;
    this.byKey = new Map(this.all.map((i) => [i.k, i]));
    this.buildFolders();
    this.emit('change');
  }

  private buildFolders(): void {
    const map = new Map<string, Folder>();
    const ensure = (path: string): Folder => {
      let f = map.get(path);
      if (!f) {
        const parts = path.split('/').filter(Boolean);
        const parent = parts.length > 1 ? parts.slice(0, -1).join('/') + '/' : '';
        f = {
          path,
          name: parts.length ? parts[parts.length - 1] : t('root'),
          parent,
          direct: [],
          children: [],
          count: 0,
          size: 0,
          latest: 0,
        };
        map.set(path, f);
        if (path !== '') ensure(parent).children.push(path);
      }
      return f;
    };
    ensure('');
    for (const it of this.items) {
      const parts = it.dir.split('/').filter(Boolean);
      ensure(it.dir).direct.push(it);
      let p = '';
      const chain = [''];
      for (const seg of parts) {
        p += seg + '/';
        chain.push(p);
      }
      for (const c of chain) {
        const f = map.get(c) ?? ensure(c);
        f.count++;
        f.size += it.size;
        if (it.taken > f.latest) {
          f.latest = it.taken;
          f.cover = it;
        }
      }
    }
    this.folders = map;
  }

  /** «Альбомы» — папки, в которых файлы лежат непосредственно */
  albums(): Folder[] {
    return [...this.folders.values()]
      .filter((f) => f.direct.length > 0)
      .sort((a, b) => b.latest - a.latest);
  }

  search(q: string): Item[] {
    const s = q.trim().toLowerCase();
    if (!s) return [];
    return this.items.filter((i) => i.name.toLowerCase().includes(s) || i.dir.toLowerCase().includes(s));
  }
}

export const store = new Store();
