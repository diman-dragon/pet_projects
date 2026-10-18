// Tiny ZIP reader (stored + deflate): returns the bytes of the first entry whose name matches. Enough for .fb2.zip.
const u16 = (dv, o) => dv.getUint16(o, true);
const u32 = (dv, o) => dv.getUint32(o, true);
async function inflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
export async function firstEntry(buf, match) {
  const dv = new DataView(buf);
  let end = buf.byteLength - 22;
  while (end >= 0 && u32(dv, end) !== 0x06054b50) end--;
  if (end < 0) throw new Error('not a zip');
  let p = u32(dv, end + 16);
  for (let n = u16(dv, end + 10); n > 0; n--) {
    if (u32(dv, p) !== 0x02014b50) break;
    const method = u16(dv, p + 10), size = u32(dv, p + 20), nameLen = u16(dv, p + 28), extra = u16(dv, p + 30), comment = u16(dv, p + 32), local = u32(dv, p + 42);
    const name = new TextDecoder().decode(new Uint8Array(buf, p + 46, nameLen));
    if (match(name)) {
      const start = local + 30 + u16(dv, local + 26) + u16(dv, local + 28);
      const data = new Uint8Array(buf, start, size);
      return method === 0 ? data : inflate(data);
    }
    p += 46 + nameLen + extra + comment;
  }
  throw new Error('no matching entry');
}
