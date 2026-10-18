// Library folder access through the native "Folder" plugin (Android Storage Access Framework).
import { plugin, nativeUrl } from '../core/platform.js';
import { acceptExts } from '../formats/registry.js';
const p = () => plugin('Folder');
export const folderSupported = () => !!p();
export async function pickFolder() {
  const r = await p().pick();
  return r?.uri ? { uri: r.uri, name: r.name || r.uri } : null;
}
export async function listFolder(uri) {
  return (await p().list({ uri, exts: acceptExts() })).files ?? [];
}
async function fetchBlob(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('read failed ' + r.status);
  return r.blob();
}
export async function readFolderFile(uri) {
  try { return await fetchBlob(nativeUrl(uri)); } catch { /* fall back to staging a copy in the app cache */ }
  const { path } = await p().stage({ uri });
  return fetchBlob(nativeUrl(path));
}
