export interface Item {
  /** уникальный ключ: i123 (фото) / v123 (видео) — id в таблицах Images и Video пересекаются */
  k: string;
  id: number;
  v: boolean;
  name: string;
  /** относительная папка, напр. "DCIM/Camera/" */
  dir: string;
  taken: number;
  added: number;
  size: number;
  w: number;
  h: number;
  mime: string;
  dur: number;
}

export interface Ref {
  id: number;
  v: boolean;
  t?: boolean;
}

export type SortKey = 'date' | 'name' | 'size';

export interface Settings {
  theme: 'auto' | 'dark' | 'light' | 'amoled';
  cols: number;
  sort: SortKey;
  desc: boolean;
  filter: 'all' | 'photo' | 'video';
  fit: boolean;
  groupByDay: boolean;
  hidden: string[];
  lang: 'auto' | 'ru' | 'en';
}

export interface Folder {
  path: string;
  name: string;
  parent: string;
  /** файлов непосредственно в папке */
  direct: Item[];
  children: string[];
  /** всего файлов, включая вложенные папки */
  count: number;
  size: number;
  latest: number;
  cover?: Item;
}

export interface Info {
  name?: string;
  dir?: string;
  path?: string;
  size?: number;
  w?: number;
  h?: number;
  mime?: string;
  taken?: number;
  dur?: number;
  make?: string;
  model?: string;
  exposure?: string;
  fnumber?: string;
  iso?: string;
  focal?: string;
  lat?: number;
  lon?: number;
}

export interface OpResult {
  done: number;
  failed: number;
  error: string;
}
