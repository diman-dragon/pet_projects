// Cover thumbnails cache (IndexedDB "covers": id -> JPEG blob).
import { tx, wait, finish } from './db.js';
export async function getCover(id) {
  const t = await tx('covers');
  return (await wait(t.objectStore('covers').get(id))) ?? null;
}
export async function putCover(id, blob) {
  const t = await tx('covers', 'readwrite');
  t.objectStore('covers').put(blob, id);
  await finish(t);
}
