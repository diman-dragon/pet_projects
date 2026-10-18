// Cover thumbnails: cached in IndexedDB, rendered one at a time in the background by the format module.
import { getCover, putCover } from '../data/covers.js';
import { find } from '../formats/registry.js';
import { updateDoc } from '../data/library.js';
import { loadBlob } from './doc-source.js';
const urls = new Map();
const bad = new Set();
const queue = [];
let busy = false;
const key = (m) => `${m.id}|${m.size}|${m.mtime ?? ''}`;
export const peekCover = (meta) => urls.get(key(meta)) ?? null;
async function make(meta) {
  let blob = await getCover(meta.id);
  if (!blob) {
    const src = await loadBlob(meta);
    const mod = src && find(meta.name, src.type);
    if (!mod?.cover) return null;
    const res = await mod.cover(src, { width: 320 });
    blob = res.blob;
    meta.pages = res.pages;
    await putCover(meta.id, blob);
    await updateDoc(meta.id, { pages: res.pages });
  }
  const url = URL.createObjectURL(blob);
  urls.set(key(meta), url);
  return url;
}
async function pump() {
  if (busy) return;
  busy = true;
  while (queue.length) {
    const job = queue.shift();
    try { job.resolve(await make(job.meta)); } catch (err) { console.warn('cover failed', job.meta.name, err); bad.add(job.meta.id); job.resolve(null); }
  }
  busy = false;
}
export function requestCover(meta) {
  if (urls.has(key(meta))) return Promise.resolve(urls.get(key(meta)));
  if (bad.has(meta.id)) return Promise.resolve(null);
  return new Promise((resolve) => { queue.push({ meta, resolve }); pump(); });
}
export function cancelCover(id) {
  const i = queue.findIndex((j) => j.meta.id === id);
  if (i >= 0) queue.splice(i, 1)[0].resolve(null);
}
