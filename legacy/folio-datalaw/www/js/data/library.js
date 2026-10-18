// Document library: records (+ bytes for files opened with "Open"; folder files are only referenced by uri).
import { tx, wait, finish } from './db.js';
import { fingerprint } from './fingerprint.js';
import { dropOcrIn } from './ocr-store.js';
import { isOpenable } from '../formats/registry.js';
export async function listDocs() {
  const t = await tx('meta');
  const all = await wait(t.objectStore('meta').getAll());
  return all.filter((d) => isOpenable(d.name)).sort((a, b) => (b.openedAt - a.openedAt) || a.name.localeCompare(b.name));
}
export async function getMeta(id) {
  const t = await tx('meta');
  return (await wait(t.objectStore('meta').get(id))) ?? null;
}
export async function getBlob(id) {
  const t = await tx('blobs');
  return (await wait(t.objectStore('blobs').get(id))) ?? null;
}
export async function addDoc(file) {
  const fp = await fingerprint(file);
  const existing = (await listDocs()).find((d) => d.src !== 'folder' && d.size === file.size && (d.fp ? d.fp === fp : d.name === file.name));
  if (existing) return updateDoc(existing.id, { openedAt: Date.now(), fp });
  const now = Date.now();
  const meta = { id: crypto.randomUUID(), name: file.name, size: file.size, fp, addedAt: now, openedAt: now, lastPage: 1 };
  const t = await tx(['meta', 'blobs'], 'readwrite');
  t.objectStore('meta').put(meta);
  t.objectStore('blobs').put(file, meta.id);
  await finish(t);
  return meta;
}
export async function updateDoc(id, patch) {
  const t = await tx('meta', 'readwrite');
  const store = t.objectStore('meta');
  const old = await wait(store.get(id));
  if (!old) return null; // deleted meanwhile (e.g. while its cover was rendering): do not recreate a broken record
  const meta = { ...old, ...patch };
  store.put(meta);
  await finish(t);
  return meta;
}
export async function removeDoc(id) {
  const t = await tx(['meta', 'blobs', 'covers', 'ocr'], 'readwrite');
  dropOcrIn(t, id);
  t.objectStore('meta').delete(id);
  t.objectStore('blobs').delete(id);
  t.objectStore('covers').delete(id);
  await finish(t);
}
