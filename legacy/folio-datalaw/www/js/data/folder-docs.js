// Mirrors a folder scan into the library: adds new files, updates changed ones, drops vanished ones (no bytes copied).
import { tx, wait, finish } from './db.js';
const idOf = (uri) => 'f:' + uri;
export async function mirrorFolder(files) {
  const t = await tx(['meta', 'covers'], 'readwrite');
  const meta = t.objectStore('meta');
  const covers = t.objectStore('covers');
  const have = new Map((await wait(meta.getAll())).filter((d) => d.src === 'folder').map((d) => [d.id, d]));
  const seen = new Set();
  for (const f of files) {
    const id = idOf(f.uri);
    const old = have.get(id);
    seen.add(id);
    if (!old) {
      meta.put({ id, src: 'folder', uri: f.uri, name: f.name, size: f.size, mtime: f.mtime, addedAt: Date.now(), openedAt: 0, lastPage: 1 });
    } else if (old.size !== f.size || old.mtime !== f.mtime || old.name !== f.name) {
      meta.put({ ...old, name: f.name, size: f.size, mtime: f.mtime, pages: 0 });
      covers.delete(id);
    }
  }
  for (const id of have.keys()) if (!seen.has(id)) { meta.delete(id); covers.delete(id); }
  await finish(t);
}
