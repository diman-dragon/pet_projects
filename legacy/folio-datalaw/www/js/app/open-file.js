// Orchestration: File -> library record -> viewer; stored record -> viewer.
import { find } from '../formats/registry.js';
import { addDoc, getMeta, updateDoc } from '../data/library.js';
import { loadBlob } from './doc-source.js';
import { openViewer } from '../ui/viewer-screen.js';
import { toast } from '../ui/toast.js';
import { t } from '../core/i18n.js';
import { emit } from '../core/bus.js';
export async function openFile(file) {
  if (!find(file.name, file.type)) { toast(t('viewer.unsupported')); return; }
  const meta = await addDoc(file);
  emit('libchange');
  await openViewer(meta, file);
}
export async function openStored(id) {
  if (!(await getMeta(id))) return;
  const meta = await updateDoc(id, { openedAt: Date.now() });
  const blob = await loadBlob(meta);
  if (!blob) return;
  emit('libchange');
  await openViewer(meta, blob);
}
