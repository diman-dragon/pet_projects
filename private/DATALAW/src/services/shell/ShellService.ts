import { open as showOpenDialog } from '@tauri-apps/plugin-dialog';
import { openPath, revealItemInDir } from '@tauri-apps/plugin-opener';
import { PathAccessService } from '../fs/PathAccessService';

export const ShellService = {
  async open(path: string): Promise<void> {
    if (!path?.trim()) {
      console.warn('ShellService: empty path');
      return;
    }

    try {
      PathAccessService.assertWorkspacePath(path);
      await openPath(path);
    } catch (error) {
      console.warn('ShellService: open failed, falling back to folder reveal', { path, error });
      try {
        await revealItemInDir(path);
      } catch (innerError) {
        console.error('ShellService: failed to open or reveal path', { path, error: innerError });
      }
    }
  },

  async selectDirectory(): Promise<string | null> {
    try {
      const selected = await showOpenDialog({
        directory: true,
        multiple: false,
        title: 'Выберите папку обращения',
      });

      if (!selected) return null;
      return Array.isArray(selected) ? selected[0] : selected;
    } catch (error) {
      console.error('ShellService: dialog error', error);
      return null;
    }
  },
};
