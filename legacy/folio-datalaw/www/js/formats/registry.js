// Registry of format modules. Module contract: { id, extensions[], mimes[], mount(ctx) -> handle, cover?(blob, opts) -> {blob, pages} }.
// Extensions may be compound ("fb2.zip"); a handle may carry caps { thumbs, draw, mode } (default: all true).
const mods = [];
export const register = (m) => { mods.push(m); };
export function find(name = '', type = '') {
  const n = name.toLowerCase();
  return mods.find((m) => m.extensions.some((e) => n.endsWith(`.${e}`)) || (type && m.mimes.includes(type))) ?? null;
}
// The library only ever shows files that can be opened.
export const isOpenable = (name, type) => !!find(name, type);
export const acceptExts = () => mods.flatMap((m) => m.extensions);
export const acceptList = () => [...new Set(mods.flatMap((m) => [...m.extensions.map((e) => `.${e.split('.').pop()}`), ...m.mimes]))].join(',');
// Book title from a file name: the (possibly compound) extension is cut off, e.g. "war.fb2.zip" -> "war".
export function titleOf(name) {
  const n = name.toLowerCase(), ext = mods.flatMap((m) => m.extensions).filter((e) => n.endsWith(`.${e}`)).sort((a, b) => b.length - a.length)[0];
  return ext ? name.slice(0, -(ext.length + 1)) : name;
}
