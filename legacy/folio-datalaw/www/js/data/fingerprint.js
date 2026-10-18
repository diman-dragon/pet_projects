// Cheap content fingerprint (SHA-1 of size + first and last 64 KB) so the same file opened twice is stored once.
const CHUNK = 65536;
export async function fingerprint(file) {
  try {
    const head = await file.slice(0, CHUNK).arrayBuffer();
    const tail = await file.slice(Math.max(0, file.size - CHUNK)).arrayBuffer();
    const all = new Uint8Array(head.byteLength + tail.byteLength + 8);
    all.set(new Uint8Array(head), 0);
    all.set(new Uint8Array(tail), head.byteLength);
    new DataView(all.buffer).setFloat64(head.byteLength + tail.byteLength, file.size);
    const h = new Uint8Array(await crypto.subtle.digest('SHA-1', all));
    return [...h].map((b) => b.toString(16).padStart(2, '0')).join('');
  } catch { return ''; }
}
