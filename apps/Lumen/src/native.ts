import { Capacitor, registerPlugin } from '@capacitor/core';
import type { Info, OpResult, Ref } from './types';

type Row = (string | number)[];

export interface MediaLibraryApi {
  checkAccess(): Promise<{ granted: boolean }>;
  requestAccess(): Promise<{ granted: boolean }>;
  scan(): Promise<{ total: number }>;
  fetch(o: { offset: number; limit: number }): Promise<{ rows: Row[] }>;
  listTrash(): Promise<{ rows: Row[] }>;
  queueThumbs(o: { items: Ref[]; size: number }): Promise<void>;
  getInfo(o: Ref): Promise<Info>;
  share(o: { items: Ref[] }): Promise<void>;
  openWith(o: Ref): Promise<void>;
  trash(o: { items: Ref[] }): Promise<OpResult>;
  restore(o: { items: Ref[] }): Promise<OpResult>;
  deleteForever(o: { items: Ref[] }): Promise<OpResult>;
  move(o: { items: Ref[]; dest: string }): Promise<OpResult>;
  copy(o: { items: Ref[]; dest: string }): Promise<OpResult>;
  rename(o: { items: Ref[]; name: string }): Promise<OpResult>;
  manageMediaStatus(): Promise<{ supported: boolean; granted: boolean }>;
  requestManageMedia(): Promise<void>;
  addListener(ev: 'thumb', fn: (e: { k: string; p: string }) => void): Promise<{ remove(): Promise<void> }>;
  addListener(ev: 'changed', fn: () => void): Promise<{ remove(): Promise<void> }>;
}

export const isNative = Capacitor.isNativePlatform();

export const api: MediaLibraryApi = isNative
  ? registerPlugin<MediaLibraryApi>('MediaLibrary')
  : createMock();

export function fullUrl(i: { id: number; v: boolean }): string {
  if (!isNative) return mockImage(i.id, 1200, 800, true);
  return Capacitor.convertFileSrc(`content://media/external/${i.v ? 'video' : 'images'}/media/${i.id}`);
}

export function fileUrl(path: string): string {
  return Capacitor.convertFileSrc(path.startsWith('/') ? `file://${path}` : path);
}

// ---------------------------------------------------------------------------
// Мок для разработки в браузере: `npm run dev`
// ---------------------------------------------------------------------------

function mockImage(id: number, w: number, h: number, large = false): string {
  const hue = (id * 47) % 360;
  const hue2 = (hue + 50) % 360;
  const label = large ? `<text x="50%" y="52%" fill="rgba(255,255,255,.85)" font-size="${w / 10}" font-family="sans-serif" text-anchor="middle">#${id}</text>` : '';
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">` +
    `<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="hsl(${hue},65%,52%)"/><stop offset="1" stop-color="hsl(${hue2},70%,32%)"/></linearGradient></defs>` +
    `<rect width="100%" height="100%" fill="url(#g)"/>` +
    `<circle cx="${w * 0.7}" cy="${h * 0.3}" r="${Math.min(w, h) * 0.1}" fill="rgba(255,255,255,.35)"/>` +
    `<path d="M0 ${h} L${w * 0.35} ${h * 0.55} L${w * 0.55} ${h * 0.8} L${w * 0.75} ${h * 0.6} L${w} ${h}Z" fill="rgba(0,0,0,.28)"/>` +
    label +
    `</svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

function createMock(): MediaLibraryApi {
  const DAY = 86400000;
  const folders: [string, number, boolean][] = [
    ['DCIM/Camera/', 1400, false],
    ['Pictures/Screenshots/', 380, false],
    ['Pictures/Telegram/', 260, false],
    ['Download/', 90, false],
    ['Pictures/Travel/Iceland/', 170, false],
    ['Pictures/Travel/Japan/', 230, false],
    ['Pictures/Family/', 120, false],
    ['Movies/', 60, true],
    ['DCIM/Camera/', 80, true],
  ];
  const rows: Row[] = [];
  const now = Date.now();
  let id = 1;
  let day = 0;
  for (const [dir, n, video] of folders) {
    let t = now - day * DAY;
    for (let i = 0; i < n; i++) {
      t -= (Math.random() * 0.8 + 0.1) * DAY * (n > 300 ? 0.4 : 1.2);
      const portrait = id % 3 === 0;
      rows.push([
        id,
        video ? 1 : 0,
        video ? `VID_${id}.mp4` : `IMG_${String(id).padStart(5, '0')}.jpg`,
        dir,
        Math.round(t),
        Math.round(t),
        500000 + ((id * 7919) % 6000000),
        portrait ? 3024 : 4032,
        portrait ? 4032 : 3024,
        video ? 'video/mp4' : 'image/jpeg',
        video ? 5000 + ((id * 313) % 120000) : 0,
      ]);
      id++;
    }
    day += 3;
  }
  rows.sort((a, b) => (b[4] as number) - (a[4] as number));

  const thumbL = new Set<(e: { k: string; p: string }) => void>();
  const changedL = new Set<() => void>();
  const ok = (n: number): OpResult => ({ done: n, failed: 0, error: '' });

  return {
    async checkAccess() {
      return { granted: true };
    },
    async requestAccess() {
      return { granted: true };
    },
    async scan() {
      return { total: rows.length };
    },
    async fetch({ offset, limit }) {
      return { rows: rows.slice(offset, offset + limit) };
    },
    async listTrash() {
      return { rows: rows.slice(0, 6) };
    },
    async queueThumbs({ items }) {
      items.forEach((it, n) =>
        setTimeout(() => {
          const p = mockImage(it.id, 240, 240);
          thumbL.forEach((f) => f({ k: (it.v ? 'v' : 'i') + it.id, p }));
        }, 10 + n * 4)
      );
    },
    async getInfo(o) {
      const r = rows.find((x) => x[0] === o.id && (x[1] === 1) === o.v);
      return {
        name: r?.[2] as string,
        dir: r?.[3] as string,
        size: r?.[6] as number,
        w: r?.[7] as number,
        h: r?.[8] as number,
        mime: r?.[9] as string,
        taken: r?.[4] as number,
        make: 'Mock',
        model: 'Pixel 9 Pro',
        exposure: '1/120',
        fnumber: '1.7',
        iso: '64',
        focal: '6.9',
        lat: 64.1466,
        lon: -21.9426,
      };
    },
    async share() {},
    async openWith() {},
    async trash({ items }) {
      return ok(items.length);
    },
    async restore({ items }) {
      return ok(items.length);
    },
    async deleteForever({ items }) {
      return ok(items.length);
    },
    async move({ items, dest }) {
      for (const it of items) {
        const r = rows.find((x) => x[0] === it.id && (x[1] === 1) === it.v);
        if (r) r[3] = dest;
      }
      return ok(items.length);
    },
    async copy({ items }) {
      return ok(items.length);
    },
    async rename({ items, name }) {
      const r = rows.find((x) => x[0] === items[0].id && (x[1] === 1) === items[0].v);
      if (r) r[2] = name;
      return ok(1);
    },
    async manageMediaStatus() {
      return { supported: true, granted: false };
    },
    async requestManageMedia() {},
    addListener(ev: string, fn: (e: never) => void) {
      if (ev === 'thumb') thumbL.add(fn as (e: { k: string; p: string }) => void);
      else changedL.add(fn as () => void);
      return Promise.resolve({ remove: async () => void 0 });
    },
  } as MediaLibraryApi;
}
