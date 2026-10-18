import { api, fileUrl, fullUrl } from './native';
import type { Item, Ref } from './types';

const SIZE = 360;

class Thumbs {
  private urls = new Map<string, string>();
  private failed = new Set<string>();
  private waiting = new Map<string, Set<HTMLImageElement>>();
  private requested = new Set<string>();
  private queue: Ref[] = [];
  private timer = 0;
  private trashMode = false;

  constructor() {
    void api.addListener('thumb', (e) => this.onThumb(e.k, e.p));
  }

  /** В режиме корзины запрашиваем миниатюры с MATCH_TRASHED. */
  setTrashMode(on: boolean): void {
    this.trashMode = on;
  }

  url(k: string): string | undefined {
    return this.urls.get(k);
  }

  attach(img: HTMLImageElement, it: Item): void {
    img.dataset.k = it.k;
    const u = this.urls.get(it.k);
    if (u) {
      this.setSrc(img, u);
      return;
    }
    if (this.failed.has(it.k)) {
      this.fallback(img, it);
      return;
    }
    let set = this.waiting.get(it.k);
    if (!set) this.waiting.set(it.k, (set = new Set()));
    set.add(img);
    if (!this.requested.has(it.k)) {
      this.requested.add(it.k);
      this.queue.push({ id: it.id, v: it.v, t: this.trashMode });
      if (!this.timer) this.timer = window.setTimeout(() => this.flush(), 24);
    }
  }

  forget(keys: Iterable<string>): void {
    for (const k of keys) {
      this.urls.delete(k);
      this.requested.delete(k);
      this.failed.delete(k);
      this.waiting.delete(k);
    }
  }

  private flush(): void {
    this.timer = 0;
    const batch = this.queue.splice(0, this.queue.length);
    if (batch.length) void api.queueThumbs({ items: batch, size: SIZE });
  }

  private onThumb(k: string, p: string): void {
    const set = this.waiting.get(k);
    this.waiting.delete(k);
    if (!p) {
      this.failed.add(k);
      this.requested.delete(k);
      set?.forEach((img) => {
        if (img.isConnected) this.fallback(img);
      });
      return;
    }
    const url = fileUrl(p);
    this.urls.set(k, url);
    set?.forEach((img) => {
      if (img.isConnected && img.dataset.k === k) this.setSrc(img, url);
    });
  }

  private setSrc(img: HTMLImageElement, url: string): void {
    img.onload = () => img.classList.add('ld');
    img.src = url;
    if (img.complete && img.naturalWidth) img.classList.add('ld');
  }

  private fallback(img: HTMLImageElement, it?: Item): void {
    const k = img.dataset.k!;
    const id = it?.id ?? Number(k.slice(1));
    const v = k[0] === 'v';
    if (v) return; // у видео без миниатюры остаётся плейсхолдер
    img.onload = () => img.classList.add('ld');
    img.src = fullUrl({ id, v });
  }
}

export const thumbs = new Thumbs();
