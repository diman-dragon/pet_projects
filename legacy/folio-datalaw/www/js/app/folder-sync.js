// Scans the chosen folder and mirrors it into the library; overlapping/rapid calls are coalesced.
import { getFolder } from '../core/folder-pref.js';
import { folderSupported, listFolder } from '../io/folder.js';
import { mirrorFolder } from '../data/folder-docs.js';
import { isOpenable } from '../formats/registry.js';
import { emit } from '../core/bus.js';
let running = null, last = 0;
export function syncFolder({ force = false } = {}) {
  if (running) return running;
  if (!force && Date.now() - last < 4000) return Promise.resolve(true);
  running = (async () => {
    const f = getFolder();
    emit('syncstate', true);
    try {
      if (!f) await mirrorFolder([]);
      else if (!folderSupported()) return false;
      else await mirrorFolder((await listFolder(f.uri)).filter((x) => isOpenable(x.name)));
      last = Date.now();
      emit('libchange');
      return true;
    } catch (err) {
      console.error(err);
      return false;
    } finally {
      running = null;
      emit('syncstate', false);
    }
  })();
  return running;
}
