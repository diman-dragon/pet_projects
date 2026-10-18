// Keeps the library in step with the folder: on start, when the app returns to the foreground, when the folder changes.
import { syncFolder } from './folder-sync.js';
import { plugin } from '../core/platform.js';
import { on } from '../core/bus.js';
export function watchFolder() {
  syncFolder({ force: true });
  on('folderchange', () => syncFolder({ force: true }));
  plugin('App')?.addListener('appStateChange', ({ isActive }) => { if (isActive) syncFolder(); });
}
