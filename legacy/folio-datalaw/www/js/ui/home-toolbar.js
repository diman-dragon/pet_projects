// Library header buttons: grid/list switch, folder refresh (spins while scanning) and "Open".
import { getLibView, setLibView } from '../core/libview.js';
import { getFolder } from '../core/folder-pref.js';
import { syncFolder } from '../app/folder-sync.js';
import { openFile } from '../app/open-file.js';
import { pickFile } from '../io/pick-file.js';
import { icon } from './icon.js';
import { toast } from './toast.js';
import { t } from '../core/i18n.js';
import { on } from '../core/bus.js';
const list = document.getElementById('doc-list');
const viewBtn = document.getElementById('view-btn');
const refreshBtn = document.getElementById('refresh-btn');
function paint() {
  const grid = getLibView() === 'grid';
  list.className = getLibView();
  viewBtn.innerHTML = icon(grid ? 'list' : 'grid');
  viewBtn.setAttribute('aria-label', t(grid ? 'home.viewList' : 'home.viewGrid'));
  refreshBtn.hidden = !getFolder();
}
viewBtn.addEventListener('click', () => setLibView(getLibView() === 'grid' ? 'list' : 'grid'));
refreshBtn.addEventListener('click', async () => { if (!(await syncFolder({ force: true }))) toast(t('folder.fail')); });
document.getElementById('open-btn').addEventListener('click', async () => {
  const file = await pickFile();
  if (file) openFile(file).catch(() => toast(t('home.pickFail')));
});
on('syncstate', (busy) => refreshBtn.classList.toggle('spin', busy));
on('viewchange', paint);
on('folderchange', paint);
on('langchange', paint);
paint();
