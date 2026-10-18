// "Open with Folio": reads the launch / new-intent URL (content://, file://) into a File.
import { plugin, nativeUrl } from '../core/platform.js';
import { isOpenable } from '../formats/registry.js';
const stamp = () => new Date().toISOString().slice(0, 16).replace('T', ' ');
// content:// links often have no file name: the real format is read from the first bytes.
async function sniff(blob) {
  const head = new Uint8Array(await blob.slice(0, 512).arrayBuffer());
  const text = new TextDecoder('latin1').decode(head);
  if (text.startsWith('%PDF')) return 'pdf';
  if (head[0] === 0x50 && head[1] === 0x4b) return 'fb2.zip';
  return /<FictionBook|<\?xml/i.test(text) ? 'fb2' : 'pdf';
}
async function nameOf(url, blob) {
  const last = decodeURIComponent(url.split('?')[0].split('/').pop() ?? '');
  return isOpenable(last) ? last : `Document ${stamp()}.${await sniff(blob)}`;
}
async function load(url, onFile) {
  const blob = await (await fetch(nativeUrl(url))).blob();
  onFile(new File([blob], await nameOf(url, blob), { type: blob.type || '' }));
}
export async function watchIntents(onFile, onError) {
  const App = plugin('App');
  if (!App) return;
  App.addListener('appUrlOpen', ({ url }) => load(url, onFile).catch(onError));
  const launch = await App.getLaunchUrl();
  if (launch?.url) load(launch.url, onFile).catch(onError);
}
