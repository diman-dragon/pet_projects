// IndexedDB connection (stores "meta" = records, "blobs" = file bytes, "covers" = thumbnails, "ocr" = recognized text) and promise helpers.
let opening;
export function db() {
  opening ??= new Promise((res, rej) => {
    const r = indexedDB.open('folio', 3);
    r.onupgradeneeded = () => {
      const d = r.result;
      if (!d.objectStoreNames.contains('meta')) d.createObjectStore('meta', { keyPath: 'id' });
      if (!d.objectStoreNames.contains('blobs')) d.createObjectStore('blobs');
      if (!d.objectStoreNames.contains('covers')) d.createObjectStore('covers');
      if (!d.objectStoreNames.contains('ocr')) d.createObjectStore('ocr');
    };
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
  return opening;
}
export const tx = async (stores, mode = 'readonly') => (await db()).transaction(stores, mode);
export const wait = (req) => new Promise((res, rej) => { req.onsuccess = () => res(req.result); req.onerror = () => rej(req.error); });
export const finish = (t) => new Promise((res, rej) => { t.oncomplete = () => res(); t.onerror = t.onabort = () => rej(t.error); });
