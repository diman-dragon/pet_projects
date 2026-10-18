// Settings row for the library folder: shows the chosen folder, choose / clear.
import { getFolder, setFolder } from '../core/folder-pref.js';
import { folderSupported, pickFolder } from '../io/folder.js';
import { toast } from './toast.js';
import { t } from '../core/i18n.js';
import { on } from '../core/bus.js';
const label = document.getElementById('folder-name');
const clear = document.getElementById('folder-clear');
function paint() {
  const f = getFolder();
  label.textContent = f ? f.name : t('set.folderNone');
  clear.hidden = !f;
}
document.getElementById('folder-pick').addEventListener('click', async () => {
  if (!folderSupported()) { toast(t('folder.unsupported')); return; }
  try {
    const f = await pickFolder();
    if (f) setFolder(f);
  } catch (err) { console.error(err); toast(t('folder.fail')); }
});
clear.addEventListener('click', () => setFolder(null));
on('folderchange', paint);
on('langchange', paint);
paint();
